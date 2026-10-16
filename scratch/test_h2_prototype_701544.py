import sys
import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from scipy.optimize import milp, LinearConstraint, Bounds

splerge_path = Path("d:/RRC/major_project_csvtu/AdvancedLiterateMachinery/DocumentUnderstanding/SPLERGE")
sys.path.insert(0, str(splerge_path))

from libs.split_model_hf import SplitModel
import libs.GTElement as gc
import libs.merge_utility as mut
import merge

def get_middle_of_groups(binary_array):
    result = np.zeros_like(binary_array)
    i = 0
    n = len(binary_array)
    while i < n:
        if binary_array[i] == 1:
            start = i
            while i < n and binary_array[i] == 1:
                i += 1
            result[(start + i - 1) // 2] = 1
        else:
            i += 1
    return result

def box_intersection_area(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    inter_x1 = max(x1, x3); inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4); inter_y2 = min(y2, y4)
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - inter
    return inter / union if union > 0 else 0.0

def count_internal_overlaps(boxes):
    overlaps = 0
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            b1, b2 = boxes[i], boxes[j]
            ix = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
            iy = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
            if ix > 0 and iy > 0:
                overlaps += 1
    return overlaps

def evaluate_matching(pred_boxes, gt_boxes, thresh=0.25):
    matched_gt = set()
    for pb in pred_boxes:
        best_iou = 0
        best_j = -1
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= thresh:
            matched_gt.add(best_j)
    tp = len(matched_gt)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    prec = tp / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
    rec = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return matched_gt, tp, fp, fn, prec, rec, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SplitModel().to(device)
    ckpt = torch.load(splerge_path / "model_out/split_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((960, 960)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_path = Path("d:/RRC/major_project_csvtu/pubtab_samples_tokens/images/pubtabnet_tok_701544.png")
    json_path = Path("d:/RRC/major_project_csvtu/pubtab_samples_tokens/iocr_jsons/pubtabnet_tok_701544_iocr.json")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_tokens = data["pages"][0].get("tokens", [])
    gt_boxes = [tok["bbox"] for tok in gt_tokens]
    raw_ocr = [[tok["id"], tok["text"], b[0], b[1], b[2], b[3]] for tok, b in zip(gt_tokens, gt_boxes)]

    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    with torch.no_grad():
        h_pred, v_pred = model(transform(img).unsqueeze(0).to(device))
        h_pred = h_pred.repeat_interleave(2, dim=1).squeeze(0).cpu().numpy()
        v_pred = v_pred.repeat_interleave(2, dim=1).squeeze(0).cpu().numpy()

    h_splits = np.where(get_middle_of_groups((h_pred > 0.5).astype(float)) == 1)[0]
    v_splits = np.where(get_middle_of_groups((v_pred > 0.5).astype(float)) == 1)[0]
    row_seps = [int(y * orig_h / 960.0) for y in h_splits if 5 < int(y * orig_h / 960.0) < orig_h - 5]
    col_seps = [int(x * orig_w / 960.0) for x in v_splits if 5 < int(x * orig_w / 960.0) < orig_w - 5]

    y_bounds = [0] + row_seps + [orig_h]
    x_bounds = [0] + col_seps + [orig_w]
    num_rows = len(row_seps) + 1
    num_cols = len(col_seps) + 1
    num_tiles = num_rows * num_cols

    print(f"\nImage: {img_path.name} ({orig_w}x{orig_h})")
    print(f"Grid Tessellation: {num_rows} rows x {num_cols} cols = {num_tiles} elementary tiles")
    print(f"Ground Truth Tokens: {len(gt_tokens)}")

    # 1. Raw Split Grid
    t_raw = gc.Table(0, 0, orig_w, orig_h)
    for r in row_seps: t_raw.gtRows.append(gc.Row(0, r, orig_w))
    for c in col_seps: t_raw.gtCols.append(gc.Column(c, 0, orig_h))
    t_raw.evaluateCells()
    raw_cells = [[c.x1, c.y1, c.x2, c.y2] for row in t_raw.gtCells for c in row if not c.dontCare]
    raw_overlaps = count_internal_overlaps(raw_cells)
    raw_m25, raw_tp25, raw_fp25, raw_fn25, raw_p25, raw_r25, raw_f1_25 = evaluate_matching(raw_cells, gt_boxes, thresh=0.25)

    # 2. Heuristic Merge Pipeline
    ocr_clean = mut.clean_ocr_data(raw_ocr)
    ocr_sentences = mut.make_sentences(ocr_clean)

    t_merge = gc.Table(0, 0, orig_w, orig_h)
    for r in row_seps: t_merge.gtRows.append(gc.Row(0, r, orig_w))
    for c in col_seps: t_merge.gtCols.append(gc.Column(c, 0, orig_h))
    t_merge.evaluateCells()
    merge.col_merging(t_merge, ocr_sentences)
    merge.row_merging(t_merge, ocr_sentences)
    merge.remove_redundant_seperators(t_merge, ocr_sentences, 0.2)
    merge.remove_redundant_seperators(t_merge, ocr_sentences, 0.4)
    t_merge.merge_header_v2(ocr_sentences)
    t_merge.evaluateCells()
    heur_cells = [[c.x1, c.y1, c.x2, c.y2] for row in t_merge.gtCells for c in row if not c.dontCare]
    heur_overlaps = count_internal_overlaps(heur_cells)
    heur_m25, heur_tp25, heur_fp25, heur_fn25, heur_p25, heur_r25, heur_f1_25 = evaluate_matching(heur_cells, gt_boxes, thresh=0.25)

    # 3. H2 Set Packing
    tile_to_idx = {(r, c): r * num_cols + c for r in range(num_rows) for c in range(num_cols)}

    candidates = []
    # Elementary
    for r in range(num_rows):
        for c in range(num_cols):
            candidates.append({
                "type": "elem",
                "r1": r, "r2": r, "c1": c, "c2": c,
                "bbox": [x_bounds[c], y_bounds[r], x_bounds[c+1], y_bounds[r+1]],
                "tiles": [(r, c)]
            })

    # Vertical spans (2, 3)
    for span_len in [2, 3]:
        for r in range(num_rows - span_len + 1):
            for c in range(num_cols):
                r_end = r + span_len - 1
                tiles = [(r_idx, c) for r_idx in range(r, r_end + 1)]
                candidates.append({
                    "type": f"v_merge_{span_len}",
                    "r1": r, "r2": r_end, "c1": c, "c2": c,
                    "bbox": [x_bounds[c], y_bounds[r], x_bounds[c+1], y_bounds[r_end+1]],
                    "tiles": tiles
                })

    # Horizontal spans (2)
    for c in range(num_cols - 1):
        for r in range(num_rows):
            tiles = [(r, c), (r, c+1)]
            candidates.append({
                "type": "h_merge_2",
                "r1": r, "r2": r, "c1": c, "c2": c+1,
                "bbox": [x_bounds[c], y_bounds[r], x_bounds[c+2], y_bounds[r+1]],
                "tiles": tiles
            })

    token_bboxes = [tok["bbox"] for tok in gt_tokens]
    token_areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in token_bboxes]

    for cand in candidates:
        cb = cand["bbox"]
        contained = []
        for t_idx, (tb, t_area) in enumerate(zip(token_bboxes, token_areas)):
            inter = box_intersection_area(tb, cb)
            if inter > 0:
                frac = inter / t_area
                contained.append((t_idx, frac))
        cand["contained"] = contained

    elem_max_frac = {}
    for t_idx in range(len(token_bboxes)):
        tb = token_bboxes[t_idx]
        t_area = token_areas[t_idx]
        max_f = 0.0
        for cand in candidates:
            if cand["type"] == "elem":
                inter = box_intersection_area(tb, cand["bbox"])
                max_f = max(max_f, inter / t_area)
        elem_max_frac[t_idx] = max_f

    weights = np.zeros(len(candidates))
    for i, cand in enumerate(candidates):
        w = 0.0
        for t, f in cand["contained"]:
            if f >= 0.70:
                w += 1.0
            elif f >= 0.50:
                w += 0.5

        if cand["type"] != "elem":
            rescued = [t for t, f in cand["contained"] if f >= 0.70 and elem_max_frac[t] < 0.70]
            if len(rescued) > 0:
                w += 1.5 * len(rescued)
            else:
                w -= 1.0
        else:
            w += 0.1

        weights[i] = max(0.001, w)

    A = np.zeros((num_tiles, len(candidates)))
    for c_idx, cand in enumerate(candidates):
        for tile in cand["tiles"]:
            u_idx = tile_to_idx[tile]
            A[u_idx, c_idx] = 1.0

    start_solve = time.perf_counter()
    c = -weights
    constraints = LinearConstraint(A, lb=np.zeros(num_tiles), ub=np.ones(num_tiles))
    integrality = np.ones(len(candidates))
    bounds = Bounds(0, 1)

    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
    solve_time_ms = (time.perf_counter() - start_solve) * 1000.0

    x_sol = np.round(res.x).astype(int)
    selected_indices = np.where(x_sol == 1)[0]
    opt_candidates = [candidates[i] for i in selected_indices]
    opt_boxes = [c["bbox"] for c in opt_candidates]

    opt_overlaps = count_internal_overlaps(opt_boxes)
    opt_m25, opt_tp25, opt_fp25, opt_fn25, opt_p25, opt_r25, opt_f1_25 = evaluate_matching(opt_boxes, gt_boxes, thresh=0.25)

    print("\n" + "="*80)
    print(f"{'Method':<30} | {'Cells':<6} | {'Overlaps':<9} | {'TP@0.25':<8} | {'Rec@0.25':<9} | {'Prec@0.25':<10}")
    print("-" * 80)
    print(f"{'1. Raw Split Grid':<30} | {len(raw_cells):<6} | {raw_overlaps:<9} | {raw_tp25:<8} | {raw_r25*100:<8.2f}% | {raw_p25*100:<9.2f}%")
    print(f"{'2. SPLERGE Heuristic Merge':<30} | {len(heur_cells):<6} | {heur_overlaps:<9} | {heur_tp25:<8} | {heur_r25*100:<8.2f}% | {heur_p25*100:<9.2f}%")
    print(f"{'3. H2 Set Packing (Ours)':<30} | {len(opt_boxes):<6} | {opt_overlaps:<9} | {opt_tp25:<8} | {opt_r25*100:<8.2f}% | {opt_p25*100:<9.2f}%")
    print("="*80)
    delta_tp = opt_tp25 - heur_tp25
    print(f"\nH2 Recovery: +{delta_tp} TPs over Heuristic Baseline")
    print(f"Solver Runtime: {solve_time_ms:.2f} ms (HiGHS Status: {res.message})")

if __name__ == "__main__":
    main()
