import sys
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp

splerge_path = Path("d:/RRC/major_project_csvtu/AdvancedLiterateMachinery/DocumentUnderstanding/SPLERGE")
sys.path.insert(0, str(splerge_path))

from libs.split_model_hf import SplitModel
import libs.GTElement as gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SplitModel().to(device)
model.load_state_dict(torch.load(splerge_path / "model_out/split_model.pth", map_location=device, weights_only=False)["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((960, 960)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
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

def run_test(weight_mode="cell_completeness"):
    img_path = Path("d:/RRC/major_project_csvtu/pubtab_samples_tokens/images/pubtabnet_tok_554187.png")
    json_path = Path("d:/RRC/major_project_csvtu/pubtab_samples_tokens/iocr_jsons/pubtabnet_tok_554187_iocr.json")
    with open(json_path) as f:
        data = json.load(f)
    gt_tokens = data["pages"][0].get("tokens", [])
    gt_boxes = [tok["bbox"] for tok in gt_tokens]

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
    tile_to_idx = {(r, c): r * num_cols + c for r in range(num_rows) for c in range(num_cols)}

    # Build Candidate Pool
    candidates = []
    # 1. Elementary
    for r in range(num_rows):
        for c in range(num_cols):
            candidates.append({
                "type": "elem",
                "r1": r, "r2": r, "c1": c, "c2": c,
                "bbox": [x_bounds[c], y_bounds[r], x_bounds[c+1], y_bounds[r+1]],
                "tiles": [(r, c)]
            })
    
    # 2. Vertical spans (length 2 and 3)
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

    # 3. Horizontal spans (length 2)
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

    # Precompute token-candidate containment
    for cand in candidates:
        cb = cand["bbox"]
        c_area = (cb[2] - cb[0]) * (cb[3] - cb[1])
        contained = [] # (tok_idx, fraction)
        for t_idx, (tb, t_area) in enumerate(zip(token_bboxes, token_areas)):
            inter = box_intersection_area(tb, cb)
            if inter > 0:
                frac = inter / t_area
                contained.append((t_idx, frac))
        cand["contained"] = contained
        cand["cell_area"] = c_area

    # Precompute for each token, its max containment in any elementary cell
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

    # Formulate weights w_c
    weights = np.zeros(len(candidates))
    for i, cand in enumerate(candidates):
        cb = cand["bbox"]
        # How many tokens are substantially inside cand?
        # A token is considered inside cand if frac >= 0.70
        tokens_inside = [t for t, f in cand["contained"] if f >= 0.70]
        # Are there tokens that straddle and are rescued by cand?
        # i.e. elem_max_frac < 0.70, but in cand frac >= 0.70!
        rescued_tokens = [t for t, f in cand["contained"] if f >= 0.70 and elem_max_frac[t] < 0.70]

        if weight_mode == "cell_completeness":
            # 1. Base value for any active cell that has content:
            # If the cell contains at least one substantial token, give base reward 1.0
            # Plus reward for each contained token
            if len(tokens_inside) > 0:
                # Base reward for being a legitimate content-bearing cell
                w = 1.0
                # Content completeness: sum of fractions of primary tokens
                for t, f in cand["contained"]:
                    if f >= 0.50:
                        w += f
                # Bonus for rescuing a fragmented token
                w += 2.0 * len(rescued_tokens)

                # Penalty if cand is a multi-tile span that merges multiple tokens
                # that were ALREADY individually complete in separate elementary cells:
                # Each individually complete token would have earned >= 1.70 on its own!
                already_complete = [t for t in tokens_inside if elem_max_frac[t] >= 0.70]
                if len(already_complete) > 1 and cand["type"] != "elem":
                    # Heavy penalty for collapsing multiple already-separated tokens into one cell
                    w -= 1.5 * (len(already_complete) - 1)
            else:
                # Empty cell: small epsilon so empty tiles can be filled if needed, or 0
                w = 0.01

        elif weight_mode == "pure_token_isolation":
            # Reward cells for holding isolated tokens
            # If cand contains 1 token: high reward
            # If cand merges multiple distinct tokens: penalize
            w = 0.0
            for t, f in cand["contained"]:
                if f >= 0.70:
                    w += 1.0
                elif f >= 0.50:
                    w += 0.5
            if cand["type"] != "elem":
                # Multi-tile candidate only gets reward if it rescues a straddling token
                if len(rescued_tokens) > 0:
                    w += 1.5 * len(rescued_tokens)
                else:
                    w -= 1.0 # discount gratuitous merges
            else:
                w += 0.1 # slight preference for elementary tessellation

        weights[i] = max(0.001, w)

    # Build constraint matrix A
    # A[u, c] = 1 if candidate c covers tile u
    # sum_{c: u in c} x_c <= 1 for each tile u
    A = np.zeros((num_tiles, len(candidates)))
    for c_idx, cand in enumerate(candidates):
        for tile in cand["tiles"]:
            u_idx = tile_to_idx[tile]
            A[u_idx, c_idx] = 1.0

    import time
    start_time = time.perf_counter()

    # Solve MILP: max w^T x <=> min -w^T x
    # subject to A x <= 1, 0 <= x <= 1, x integer
    c = -weights
    constraints = LinearConstraint(A, lb=np.zeros(num_tiles), ub=np.ones(num_tiles))
    integrality = np.ones(len(candidates))
    bounds = Bounds(0, 1)

    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
    solve_time_ms = (time.perf_counter() - start_time) * 1000.0

    print(f"\nMode: {weight_mode}")
    print(f"Solver success: {res.success}, status: {res.status}, message: {res.message}")
    print(f"Solver runtime: {solve_time_ms:.2f} ms")

    x_sol = np.round(res.x).astype(int)
    selected_indices = np.where(x_sol == 1)[0]
    selected_candidates = [candidates[i] for i in selected_indices]
    selected_boxes = [c["bbox"] for c in selected_candidates]

    print(f"Selected candidates count: {len(selected_candidates)} (out of {len(candidates)})")
    
    # Types of selected candidates
    type_counts = {}
    for c in selected_candidates:
        type_counts[c["type"]] = type_counts.get(c["type"], 0) + 1
    print(f"Selected types breakdown: {type_counts}")

    # Overlap check
    overlaps = count_internal_overlaps(selected_boxes)
    print(f"Internal overlaps among selected cells: {overlaps} (Expected: 0)")

    # Tile coverage check: how many elementary tiles are covered?
    covered_tiles = sum(len(c["tiles"]) for c in selected_candidates)
    print(f"Covered tiles: {covered_tiles} / {num_tiles} ({covered_tiles/num_tiles*100:.1f}%)")

    # Evaluation against GT
    m50, tp50, fp50, fn50, p50, r50, f1_50 = evaluate_matching(selected_boxes, gt_boxes, thresh=0.50)
    m25, tp25, fp25, fn25, p25, r25, f1_25 = evaluate_matching(selected_boxes, gt_boxes, thresh=0.25)

    print(f"\n--- Evaluation Results ---")
    print(f"IoU >= 0.50: TP={tp50}, FP={fp50}, FN={fn50}, Prec={p50*100:.2f}%, Rec={r50*100:.2f}%, F1={f1_50*100:.2f}%")
    print(f"IoU >= 0.25: TP={tp25}, FP={fp25}, FN={fn25}, Prec={p25*100:.2f}%, Rec={r25*100:.2f}%, F1={f1_25*100:.2f}%")

    # Check flagship tokens:
    # tok 9: PCa-004
    # tok 10: AY341247.1
    # tok 12: PCa-005
    # tok 13: AC104771.4
    flagship_ids = [9, 10, 12, 13]
    print("\n--- Flagship Cells Check ---")
    for fid in flagship_ids:
        tok = gt_tokens[fid]
        tb = tok["bbox"]
        best_iou, best_c = 0.0, None
        for c in selected_candidates:
            iou = compute_iou(c["bbox"], tb)
            if iou > best_iou:
                best_iou = iou
                best_c = c
        is_matched = fid in m25
        print(f"Token #{fid} '{tok['text']}': Matched={is_matched}, Best IoU={best_iou:.3f}, Cell={best_c['type']} r=[{best_c['r1']},{best_c['r2']}] c=[{best_c['c1']},{best_c['c2']}] bbox={best_c['bbox']}")

if __name__ == "__main__":
    print("=== Testing Weight Mode 1: cell_completeness ===")
    run_test("cell_completeness")
    print("\n" + "="*50 + "\n")
    print("=== Testing Weight Mode 2: pure_token_isolation ===")
    run_test("pure_token_isolation")
