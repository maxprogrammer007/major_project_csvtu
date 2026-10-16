import sys
import os
import json
import torch
import cv2
import numpy as np
from pathlib import Path

splerge_path = Path("d:/RRC/major_project_csvtu/AdvancedLiterateMachinery/DocumentUnderstanding/SPLERGE")
sys.path.insert(0, str(splerge_path))

from libs.model import SplitModel
import libs.utils as utils
import libs.GTElement as gc

def box_intersection_area(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    if ix2 < ix1 or iy2 < iy1: return 0.0
    return (ix2 - ix1) * (iy2 - iy1)

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    inter_x1 = max(x1, x3); inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4); inter_y2 = min(y2, y4)
    if inter_x2 < inter_x1 or inter_y2 < inter_y1: return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - inter
    return inter / union if union > 0 else 0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading ORIGINAL SplitModel on {device}...")
model = SplitModel(eval_mode=True).to(device)
ckpt = torch.load(splerge_path / "model_out/split_model.pth", map_location=device, weights_only=False)
try:
    model.load_state_dict(ckpt["model_state_dict"])
    print("Loaded weights successfully!")
except Exception as e:
    print(f"Error loading weights into libs.model.SplitModel: {e}")
    sys.exit(1)

model.eval()

img_path = Path("d:/RRC/major_project_csvtu/pubtab_samples_tokens/images/pubtabnet_tok_554187.png")
json_path = Path("d:/RRC/major_project_csvtu/pubtab_samples_tokens/iocr_jsons/pubtabnet_tok_554187_iocr.json")

with open(json_path, "r") as f:
    gt_data = json.load(f)

gt_tokens = gt_data["pages"][0].get("tokens", [])
gt_boxes = [tok["bbox"] for tok in gt_tokens]
print(f"Table 554187 has {len(gt_boxes)} GT cells.")

image = cv2.imread(str(img_path)).astype("float32")
H, W, C = image.shape
resized_image = utils.resize_image(image).transpose((2, 0, 1))
input_image = utils.normalize_numpy_image(resized_image).unsqueeze(0)

print(f"Original shape: {H}x{W}. Resized input shape: {input_image.shape}")

with torch.no_grad():
    rpn_out, cpn_out = model(input_image.to(device))

rpn_image = utils.probs_to_image(rpn_out.detach().clone(), input_image.shape, 1).cpu()
cpn_image = utils.probs_to_image(cpn_out.detach().clone(), input_image.shape, 0).cpu()

grid_img, row_image, col_image = utils.binary_grid_from_prob_images(rpn_image, cpn_image)
row_np_image = utils.tensor_to_numpy_image(row_image)
col_np_image = utils.tensor_to_numpy_image(col_image)

# Project back to original dimensions
row_seps = utils.get_column_separators(row_np_image, is_row=True)
col_seps = utils.get_column_separators(col_np_image, is_row=False)

# The row_np_image is sized to the input_image shape. 
# It was resized by some scale. We need to find the scale to map back to original H, W
scale1 = 600 / min(H, W)
scale2 = 1024 / max(H, W)
scale = min(scale1, scale2)

row_seps_orig = [int(y / scale) for y in row_seps]
col_seps_orig = [int(x / scale) for x in col_seps]

print(f"Row seps (orig): {row_seps_orig}")
print(f"Col seps (orig): {col_seps_orig}")

# Build Raw grid
t_raw = gc.Table(0, 0, W, H)
for r in row_seps_orig: t_raw.gtRows.append(gc.Row(0, r, W))
for c in col_seps_orig: t_raw.gtCols.append(gc.Col(0, c, H))

t_raw.rows = sorted([r.y for r in t_raw.gtRows])
t_raw.cols = sorted([c.x for c in t_raw.gtCols])
y_bounds = [0] + t_raw.rows + [H]
x_bounds = [0] + t_raw.cols + [W]

raw_cells = []
for i in range(len(y_bounds)-1):
    for j in range(len(x_bounds)-1):
        raw_cells.append([x_bounds[j], y_bounds[i], x_bounds[j+1], y_bounds[i+1]])

# Eval raw
matched_gt = set()
for pb in raw_cells:
    best_iou = 0
    best_j = -1
    for j, gb in enumerate(gt_boxes):
        if j in matched_gt: continue
        iou = compute_iou(pb, gb)
        if iou > best_iou:
            best_iou = iou
            best_j = j
    if best_iou >= 0.25:
        matched_gt.add(best_j)

tp = len(matched_gt)
print(f"Raw Recall @0.25: {tp}/{len(gt_boxes)} ({tp/len(gt_boxes)*100:.2f}%)")
