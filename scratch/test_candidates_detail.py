import sys
import json
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

def main():
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

    print(f"Table dimensions: {num_rows} rows x {num_cols} cols")
    num_tiles = num_rows * num_cols

    # Build Candidate Pool C:
    # 1. All elementary cells (r, r, c, c)
    # 2. Vertical merge spans (r1, r2, c, c) for r2 - r1 in [1, 2]
    # 3. Horizontal merge spans (r, r, c1, c2) for c2 - c1 in [1, 2]
    candidates = []
    
    # 1. Elementary
    for r in range(num_rows):
        for c in range(num_cols):
            candidates.append({
                "type": "elem",
                "r1": r, "r2": r,
                "c1": c, "c2": c,
                "bbox": [x_bounds[c], y_bounds[r], x_bounds[c+1], y_bounds[r+1]],
                "tiles": [(r, c)]
            })
    
    elem_count = len(candidates)

    # 2. Vertical spans of length 2 and 3
    for span_len in [2, 3]:
        for r in range(num_rows - span_len + 1):
            for c in range(num_cols):
                r_end = r + span_len - 1
                tiles = [(r_idx, c) for r_idx in range(r, r_end + 1)]
                candidates.append({
                    "type": f"v_merge_{span_len}",
                    "r1": r, "r2": r_end,
                    "c1": c, "c2": c,
                    "bbox": [x_bounds[c], y_bounds[r], x_bounds[c+1], y_bounds[r_end+1]],
                    "tiles": tiles
                })

    # 3. Horizontal spans of length 2
    for c in range(num_cols - 1):
        for r in range(num_rows):
            tiles = [(r, c), (r, c+1)]
            candidates.append({
                "type": "h_merge_2",
                "r1": r, "r2": r,
                "c1": c, "c2": c+1,
                "bbox": [x_bounds[c], y_bounds[r], x_bounds[c+2], y_bounds[r+1]],
                "tiles": tiles
            })

    print(f"Total candidates generated: {len(candidates)} ({elem_count} elementary, {len(candidates) - elem_count} merge spans)")

    # Test candidate weighting formulation
    # Let's inspect how tokens interact with each candidate
    token_bboxes = [tok["bbox"] for tok in gt_tokens]
    token_areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in token_bboxes]

    # For each candidate, compute:
    # - tokens with containment >= 0.70 (contained)
    # - tokens with containment in [0.20, 0.70) (partial / straddling)
    # - empty penalty or base reward
    for cand in candidates:
        cb = cand["bbox"]
        c_area = (cb[2] - cb[0]) * (cb[3] - cb[1])
        contained_tokens = []
        partial_tokens = []
        total_token_content_area = 0.0

        for t_idx, (tb, t_area) in enumerate(zip(token_bboxes, token_areas)):
            inter = box_intersection_area(tb, cb)
            if inter > 0:
                frac = inter / t_area
                if frac >= 0.70:
                    contained_tokens.append((t_idx, frac))
                    total_token_content_area += inter
                elif frac >= 0.20:
                    partial_tokens.append((t_idx, frac))
                    total_token_content_area += inter

        cand["contained_tokens"] = contained_tokens
        cand["partial_tokens"] = partial_tokens
        cand["content_area"] = total_token_content_area
        cand["cell_area"] = c_area

    # Check the 4 flagship cells:
    # PCa-004 is tok #9
    # AY341247.1 is tok #10
    # PCa-005 is tok #12
    # AC104771.4 is tok #13
    print("\n--- Inspecting Candidates for PCa-004 & PCa-005 ---")
    for cand in candidates:
        r1, r2, c1, c2 = cand["r1"], cand["r2"], cand["c1"], cand["c2"]
        if c1 == 0 and c2 == 0:
            if (r1 == 3 and r2 == 3) or (r1 == 4 and r2 == 4) or (r1 == 3 and r2 == 4):
                print(f"Cand {cand['type']} r=[{r1}, {r2}], bbox={cand['bbox']}:")
                print(f"   contained tokens: {cand['contained_tokens']}")
                print(f"   partial tokens: {cand['partial_tokens']}")

if __name__ == "__main__":
    main()
