import sys
import os
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
sys.path.insert(0, 'd:/RRC/major_project_csvtu')

from libs.split_model_hf import SplitModel
import libs.GTElement as gc
import libs.merge_utility as mut
import merge
def get_middle_of_groups(binary_array):
    result = np.zeros_like(binary_array)
    i = 0
    n = binary_array.size
    while i < n:
        if binary_array[i] == 1:
            start = i
            while i < n and binary_array[i] == 1:
                i += 1
            result[(start + i - 1) // 2] = 1
        else:
            i += 1
    return result

def generate_grid_tiles(orig_w, orig_h, row_seps, col_seps):
    y_bounds = [0] + row_seps + [orig_h]
    x_bounds = [0] + col_seps + [orig_w]
    num_rows = len(y_bounds) - 1
    num_cols = len(x_bounds) - 1
    
    grid_tiles = {}
    for r in range(num_rows):
        for c in range(num_cols):
            grid_tiles[(r, c)] = [x_bounds[c], y_bounds[r], x_bounds[c+1], y_bounds[r+1]]
    return grid_tiles, num_rows, num_cols


def box_intersection_area(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)

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
    return matched_gt

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

def analyze_table(table_id):
    print(f"\nAnalyzing {table_id}...")
    img_path = Path(f"d:/RRC/major_project_csvtu/pubtab_samples_tokens/images/pubtabnet_tok_{table_id}.png")
    gt_path = Path(f"d:/RRC/major_project_csvtu/pubtab_samples_tokens/iocr_jsons/pubtabnet_tok_{table_id}_iocr.json")
    
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    
    gt_tokens = gt_data["pages"][0].get("tokens", [])
    gt_boxes = [tok["bbox"] for tok in gt_tokens]
    
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        r_pred, c_pred = model(img_tensor)
    
    r_pred = (torch.sigmoid(r_pred) > 0.5).squeeze().cpu().numpy()
    c_pred = (torch.sigmoid(c_pred) > 0.5).squeeze().cpu().numpy()
    
    row_seps = get_middle_of_groups(r_pred)
    col_seps = get_middle_of_groups(c_pred)
    row_seps = [int(x * orig_h / 960) for x in np.where(row_seps == 1)[0]]
    col_seps = [int(x * orig_w / 960) for x in np.where(col_seps == 1)[0]]
    
    grid_tiles, num_rows, num_cols = generate_grid_tiles(orig_w, orig_h, row_seps, col_seps)
    
    for gt_idx, gb in enumerate(gt_boxes):
        span_r_min, span_r_max = float('inf'), -float('inf')
        span_c_min, span_c_max = float('inf'), -float('inf')
        
        has_overlap = False
        for (r, c), tile in grid_tiles.items():
            inter = box_intersection_area(gb, tile)
            tile_area = (tile[2] - tile[0]) * (tile[3] - tile[1])
            gt_area = (gb[2] - gb[0]) * (gb[3] - gb[1])
            
            if inter / gt_area > 0.05 or inter / tile_area > 0.1:
                span_r_min = min(span_r_min, r)
                span_r_max = max(span_r_max, r)
                span_c_min = min(span_c_min, c)
                span_c_max = max(span_c_max, c)
                has_overlap = True
                
        if has_overlap:
            height = span_r_max - span_r_min + 1
            width = span_c_max - span_c_min + 1
            if height > 3 or width > 2:
                print(f"GT Cell {gt_idx} {gb} exceeds cap: requires height={height}, width={width}")
                
    print(f"Table {table_id}: {len(gt_boxes)} GT cells.")


analyze_table('586722')
analyze_table('711083')
