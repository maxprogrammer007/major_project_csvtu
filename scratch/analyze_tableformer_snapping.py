import json
from pathlib import Path
import numpy as np

def compute_box_iou(b1, b2):
    x0, y0 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x1, y1 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter_area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0: return 0.0
    return inter_area / union_area

def match_boxes(pred_boxes, gt_boxes, iou_thresh=0.5):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return []
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=float)
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_box_iou(pb, gb)
    
    matches = []
    matched_preds = set()
    matched_gts = set()
    flat_indices = np.argsort(iou_matrix.ravel())[::-1]
    for idx in flat_indices:
        iou_val = iou_matrix.ravel()[idx]
        if iou_val < iou_thresh: break
        p_idx, g_idx = int(idx // len(gt_boxes)), int(idx % len(gt_boxes))
        if p_idx not in matched_preds and g_idx not in matched_gts:
            matched_preds.add(p_idx)
            matched_gts.add(g_idx)
            matches.append((p_idx, g_idx, float(iou_val)))
    return matches

with open("scratch/tableformer_100_tables_cache.json", "r") as f:
    cache = json.load(f)

print(f"Loaded {len(cache)} tables from TableFormer cache.")

total_gt = 0
total_raw_pred = 0
total_raw_tp = 0
total_post_pred = 0
total_post_tp = 0

for entry in cache:
    gt = entry["gt_boxes"]
    raw = entry["raw_boxes"]
    post = entry["post_boxes"]
    
    total_gt += len(gt)
    total_raw_pred += len(raw)
    total_post_pred += len(post)
    
    m_raw = match_boxes(raw, gt, 0.5)
    m_post = match_boxes(post, gt, 0.5)
    
    total_raw_tp += len(m_raw)
    total_post_tp += len(m_post)

print(f"Total GT cells:       {total_gt}")
print(f"Total Raw Preds:      {total_raw_pred} | Raw TPs (@0.5):  {total_raw_tp} ({total_raw_tp/total_gt:.2%})")
print(f"Total Post Preds:     {total_post_pred} | Post TPs (@0.5): {total_post_tp} ({total_post_tp/total_gt:.2%})")

# Let's inspect coordinate differences between raw and post
# For cells that exist in post, what was their original raw box?
print("\nInspecting coordinate snapping on table 0:")
t0 = cache[0]
print(f"Table {t0['stem']}: {len(t0['gt_boxes'])} GT, {len(t0['raw_boxes'])} Raw, {len(t0['post_boxes'])} Post")
exact_matches_gt = sum(1 for p in t0['post_boxes'] if p in t0['gt_boxes'])
print(f"Post boxes identical to GT boxes: {exact_matches_gt} / {len(t0['post_boxes'])} ({exact_matches_gt/len(t0['post_boxes']):.1%})")
exact_matches_raw = sum(1 for r in t0['raw_boxes'] if r in t0['gt_boxes'])
print(f"Raw boxes identical to GT boxes:  {exact_matches_raw} / {len(t0['raw_boxes'])} ({exact_matches_raw/len(t0['raw_boxes']):.1%})")

