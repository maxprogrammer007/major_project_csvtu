"""
Extend H2 Set Packing (Maximum Weight Independent Set) to Paradigm 2: Object Detection (TATR).
This script replaces TATR's greedy heuristic NMS with exact MWIS solved via HiGHS.
"""

import os
import sys
import copy
import json
import csv
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
from PIL import Image
from transformers import TableTransformerForObjectDetection, DetrImageProcessor
from fitz import Rect

import tatr_postprocess as postprocess

def compute_box_iou(b1, b2):
    x0 = max(b1[0], b2[0]); y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2]); y1 = min(b1[3], b2[3])
    inter_w = max(0.0, x1 - x0)
    inter_h = max(0.0, y1 - y0)
    inter_area = inter_w * inter_h
    area1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union_area = area1 + area2 - inter_area
    if union_area <= 0: return 0.0
    return inter_area / union_area

def vertical_overlap_ratio(r1, r2):
    # Not used anymore
    pass

def horizontal_overlap_ratio(c1, c2):
    # Not used anymore
    pass

def object2_overlap(b1, b2):
    """
    Returns (overlap_ratio_b1, overlap_ratio_b2)
    where overlap_ratio_b1 = Area(b1 \cap b2) / Area(b1)
    """
    x0 = max(b1[0], b2[0]); y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2]); y1 = min(b1[3], b2[3])
    inter_w = max(0.0, x1 - x0)
    inter_h = max(0.0, y1 - y0)
    inter_area = inter_w * inter_h
    
    area1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    
    r1 = inter_area / area1 if area1 > 0 else 0
    r2 = inter_area / area2 if area2 > 0 else 0
    return r1, r2

def match_boxes(pred_boxes, gt_boxes, iou_thresh=0.5):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return [], set(range(len(pred_boxes))), set(range(len(gt_boxes)))
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
        p_idx = int(idx // len(gt_boxes))
        g_idx = int(idx % len(gt_boxes))
        if p_idx not in matched_preds and g_idx not in matched_gts:
            matched_preds.add(p_idx)
            matched_gts.add(g_idx)
            matches.append((p_idx, g_idx, float(iou_val)))
    unmatched_preds = set(range(len(pred_boxes))) - matched_preds
    unmatched_gts = set(range(len(gt_boxes))) - matched_gts
    return matches, unmatched_preds, unmatched_gts

def solve_mwis(candidates, overlap_func, threshold=0.05):
    """
    Solve Maximum Weight Independent Set for a list of object dicts using HiGHS.
    candidates: list of dicts with 'bbox' and 'score'.
    overlap_func: function to compute overlap ratio between two bboxes.
    threshold: if overlap > threshold, they conflict.
    """
    if len(candidates) <= 1:
        return candidates, 0.0

    n = len(candidates)
    weights = np.array([c['score'] for c in candidates])
    
    # Build conflict edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            r1, r2 = overlap_func(candidates[i]['bbox'], candidates[j]['bbox'])
            if r1 > threshold or r2 > threshold:
                edges.append((i, j))
                
    if len(edges) == 0:
        return candidates, 0.0
        
    row_ind = []
    col_ind = []
    data_vals = []
    for e_idx, (u, v) in enumerate(edges):
        row_ind.extend([e_idx, e_idx])
        col_ind.extend([u, v])
        data_vals.extend([1.0, 1.0])
        
    A_sparse = sp.csc_matrix((data_vals, (row_ind, col_ind)), shape=(len(edges), n))
    c = -weights  # Scipy minimizes c^T x
    constraints = LinearConstraint(A_sparse, lb=np.zeros(len(edges)), ub=np.ones(len(edges)))
    integrality = np.ones(n)
    bounds = Bounds(0, 1)
    
    start_time = time.perf_counter()
    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
    solve_time_ms = (time.perf_counter() - start_time) * 1000.0
    
    if not res.success:
        print("MILP failed, returning greedy NMS fallback")
        return candidates, solve_time_ms
        
    x_sol = np.round(res.x).astype(int)
    selected_indices = np.where(x_sol == 1)[0]
    opt_candidates = [candidates[i] for i in selected_indices]
    
    # Preserve original TATR sorting
    if 'y' in overlap_func.__name__ or 'vert' in overlap_func.__name__:
        opt_candidates = sorted(opt_candidates, key=lambda k: k['bbox'][1] + k['bbox'][3])
    else:
        opt_candidates = sorted(opt_candidates, key=lambda k: k['bbox'][0] + k['bbox'][2])
        
    return opt_candidates, solve_time_ms


def main():
    images_dir = Path("d:/RRC/major_project_csvtu/pubtables_test_sample/images")
    ann_dir = Path("d:/RRC/major_project_csvtu/pubtables_test_sample/annotations")
    out_csv = Path("d:/RRC/major_project_csvtu/tatr_h2_per_table_metrics.csv")
    out_json = Path("d:/RRC/major_project_csvtu/tatr_h2_audit_summary.json")
    
    image_paths = sorted(images_dir.glob("*.jpg"))
    print(f"Total tables to evaluate: {len(image_paths)}")
    
    model_id = "microsoft/table-transformer-structure-recognition-v1.1-all"
    print(f"Loading {model_id}...")
    processor = DetrImageProcessor.from_pretrained(model_id)
    model = TableTransformerForObjectDetection.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    SCORE_THRESH = 0.5
    SIZE = {"shortest_edge": 800, "longest_edge": 1000}
    
    records = []
    runtimes = []
    
    for idx, img_path in enumerate(image_paths, 1):
        xml_path = ann_dir / (img_path.stem + ".xml")
        if not xml_path.exists(): continue
            
        tree = ET.parse(xml_path)
        root = tree.getroot()
        gt_rows = []
        gt_cols = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            b = obj.find("bndbox")
            bbox = [float(b.find(k).text) for k in ["xmin", "ymin", "xmax", "ymax"]]
            if name == "table row": gt_rows.append(bbox)
            elif name == "table column": gt_cols.append(bbox)
                
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt", size=SIZE)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        target_sizes = torch.tensor([img.size[::-1]]).to(device)
        results = processor.post_process_object_detection(outputs, threshold=SCORE_THRESH, target_sizes=target_sizes)[0]
        
        labels = results["labels"].tolist()
        scores = results["scores"].tolist()
        boxes = results["boxes"].tolist()
        
        raw_rows = []
        raw_cols = []
        for b, s, l in zip(boxes, scores, labels):
            lbl_name = model.config.id2label[l]
            obj_dict = {"bbox": [float(coord) for coord in b], "score": float(s), "label": l}
            if lbl_name == "table row": raw_rows.append(obj_dict)
            elif lbl_name == "table column": raw_cols.append(obj_dict)
                
        # Heuristic NMS
        heur_rows = postprocess.refine_rows(copy.deepcopy(raw_rows), tokens=[], score_threshold=SCORE_THRESH)
        heur_cols = postprocess.refine_columns(copy.deepcopy(raw_cols), tokens=[], score_threshold=SCORE_THRESH)
        
        # H2 MWIS
        opt_rows, rt_row = solve_mwis(copy.deepcopy(raw_rows), object2_overlap, threshold=0.5)
        opt_cols, rt_col = solve_mwis(copy.deepcopy(raw_cols), object2_overlap, threshold=0.25)
        runtimes.append(rt_row + rt_col)
        
        # Align rows and columns (NMS)
        row_rect_h = Rect(); col_rect_h = Rect()
        for obj in heur_rows: row_rect_h.include_rect(obj["bbox"])
        for obj in heur_cols: col_rect_h.include_rect(obj["bbox"])
        table_bbox_h = [col_rect_h[0], row_rect_h[1], col_rect_h[2], row_rect_h[3]]
        heur_cols = postprocess.align_columns(heur_cols, table_bbox_h)
        heur_rows = postprocess.align_rows(heur_rows, table_bbox_h)
        
        # Align rows and columns (H2)
        row_rect_opt = Rect(); col_rect_opt = Rect()
        for obj in opt_rows: row_rect_opt.include_rect(obj["bbox"])
        for obj in opt_cols: col_rect_opt.include_rect(obj["bbox"])
        table_bbox_opt = [col_rect_opt[0], row_rect_opt[1], col_rect_opt[2], row_rect_opt[3]]
        opt_cols = postprocess.align_columns(opt_cols, table_bbox_opt)
        opt_rows = postprocess.align_rows(opt_rows, table_bbox_opt)
        
        # Eval Heuristic
        heur_row_bboxes = [c["bbox"] for c in heur_rows]
        matches_h, _, fn_h = match_boxes(heur_row_bboxes, gt_rows, iou_thresh=0.5)
        heur_tp = len(matches_h)
        heur_fn = len(fn_h)
        
        # Eval H2 MWIS
        opt_row_bboxes = [c["bbox"] for c in opt_rows]
        matches_o, _, fn_o = match_boxes(opt_row_bboxes, gt_rows, iou_thresh=0.5)
        opt_tp = len(matches_o)
        opt_fn = len(fn_o)
        
        # Check raw
        raw_row_bboxes = [c["bbox"] for c in raw_rows]
        matches_r, _, fn_r = match_boxes(raw_row_bboxes, gt_rows, iou_thresh=0.5)
        raw_tp = len(matches_r)
        
        destroyed_by_heur = max(0, raw_tp - heur_tp)
        recovered_by_opt = max(0, opt_tp - heur_tp)
        
        records.append({
            "table_id": img_path.stem,
            "gt_rows": len(gt_rows),
            "raw_rows": len(raw_rows),
            "raw_tp": raw_tp,
            "heur_rows": len(heur_rows),
            "heur_tp": heur_tp,
            "opt_rows": len(opt_rows),
            "opt_tp": opt_tp,
            "destroyed_by_heur": destroyed_by_heur,
            "recovered_by_opt": recovered_by_opt
        })
        
        if idx % 10 == 0:
            print(f"Processed {idx}/{len(image_paths)}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    total_gt = sum(r["gt_rows"] for r in records)
    total_raw_tp = sum(r["raw_tp"] for r in records)
    total_heur_tp = sum(r["heur_tp"] for r in records)
    total_opt_tp = sum(r["opt_tp"] for r in records)
    
    destroyed = max(0, total_raw_tp - total_heur_tp)
    recovered = max(0, total_opt_tp - total_heur_tp)
    
    summary = {
        "tables_evaluated": len(records),
        "total_gt_rows": total_gt,
        "raw_tp": total_raw_tp,
        "heur_tp": total_heur_tp,
        "opt_tp": total_opt_tp,
        "heur_destroyed": destroyed,
        "opt_recovered": recovered,
        "mean_solve_ms": float(np.mean(runtimes)),
        "max_solve_ms": float(np.max(runtimes)),
    }
    
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\nOptimization Summary:")
    print(json.dumps(summary, indent=2))
    
if __name__ == "__main__":
    main()
