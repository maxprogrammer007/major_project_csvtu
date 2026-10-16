import json
from pathlib import Path
import sys

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

def generate_grid_tiles(y_cuts, x_cuts, orig_h, orig_w):
    row_seps = y_cuts
    col_seps = x_cuts
    y_bounds = [0] + row_seps + [orig_h]
    x_bounds = [0] + col_seps + [orig_w]
    num_rows = len(y_bounds) - 1
    num_cols = len(x_bounds) - 1
    
    grid_tiles = {}
    for r in range(num_rows):
        for c in range(num_cols):
            y1_b, y2_b = y_bounds[r], y_bounds[r+1]
            x1_b, x2_b = x_bounds[c], x_bounds[c+1]
            grid_tiles[(r, c)] = [x1_b, y1_b, x2_b, y2_b]
            
    return grid_tiles, num_rows, num_cols

def analyze(table_id):
    gt_path = Path(f'd:/RRC/major_project_csvtu/pubtab_samples_tokens/iocr_jsons/pubtabnet_tok_{table_id}_iocr.json')
    raw_output_path = Path(f'd:/RRC/major_project_csvtu/splerge_outputs/pubtabnet_tok_{table_id}_iocr_raw_split.json')
    
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
        
    with open(raw_output_path, 'r') as f:
        raw_output = json.load(f)
        
    y_cuts = raw_output['row_cuts']
    x_cuts = raw_output['col_cuts']
    
    orig_w, orig_h = 0, 0
    if 'img' in gt_data:
        orig_w = gt_data['img']['width']
        orig_h = gt_data['img']['height']
    else:
        orig_w = gt_data['img_width']
        orig_h = gt_data['img_height']
        
    grid_tiles, num_rows, num_cols = generate_grid_tiles(y_cuts, x_cuts, orig_h, orig_w)
    
    gt_boxes = []
    if 'html' in gt_data and 'cells' in gt_data['html']:
        for cell in gt_data['html']['cells']:
            if 'bbox' in cell:
                gt_boxes.append(cell['bbox'])
    
    print(f"Table {table_id}: {len(gt_boxes)} GT cells")
    
    # We want to find GT cells that take more than 3 rows or 2 cols
    # For each GT box, find the span of tiles that have IOU > 0 or intersection area > 0.5 * tile_area
    for gt_idx, gt_box in enumerate(gt_boxes):
        span_r_min, span_r_max = float('inf'), -float('inf')
        span_c_min, span_c_max = float('inf'), -float('inf')
        
        has_overlap = False
        for (r, c), tile in grid_tiles.items():
            # compute intersection
            ix1 = max(gt_box[0], tile[0]); iy1 = max(gt_box[1], tile[1])
            ix2 = min(gt_box[2], tile[2]); iy2 = min(gt_box[3], tile[3])
            
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                tile_area = (tile[2] - tile[0]) * (tile[3] - tile[1])
                gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                
                # If it's a significant overlap
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
                print(f"GT Box {gt_idx} {gt_box} requires height={height}, width={width}")

analyze('586722')
analyze('711083')
