"""
Measure structural consistency, catastrophic collapse, and repair fidelity
for Microsoft Table Transformer (TATR-v1.1-All) on PubTables-1M test set.

Paradigm 2: Object Detection / DETR (vs Paradigm 1: Sequence Generation / TableFormer).

Investigates:
1. Raw structural consistency: Do DETR row/column predictions overlap?
2. Catastrophic collapse: Is there a heavy-tailed failure mode analogous to TableFormer?
3. Repair audit: How does refine_rows/columns + align_rows/columns affect geometry?
4. Structural accuracy: Bipartite IoU matching against PubTables-1M ground truth (no circularity).
5. Over-pruning: Does heuristic NMS discard valid rows or columns?
"""

import os
import sys
import copy
import json
import csv
import math
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import TableTransformerForObjectDetection, DetrImageProcessor
from fitz import Rect

import tatr_postprocess as postprocess

def compute_box_iou(b1, b2):
    """Compute IoU between two bboxes [x0, y0, x1, y1]."""
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2])
    y1 = min(b1[3], b2[3])
    
    inter_w = max(0.0, x1 - x0)
    inter_h = max(0.0, y1 - y0)
    inter_area = inter_w * inter_h
    
    area1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def boxes_overlap(b1, b2):
    """Standard 2D bounding box intersection check."""
    if b1[0] >= b2[2] or b1[2] <= b2[0] or b1[3] <= b2[1] or b1[1] >= b2[3]:
        return False
    return True

def vertical_overlap_ratio(r1, r2):
    """1D vertical overlap relative to smaller row height."""
    inter = max(0.0, min(r1[3], r2[3]) - max(r1[1], r2[1]))
    h1 = max(1e-5, r1[3] - r1[1])
    h2 = max(1e-5, r2[3] - r2[1])
    return inter / min(h1, h2)

def horizontal_overlap_ratio(c1, c2):
    """1D horizontal overlap relative to smaller col width."""
    inter = max(0.0, min(c1[2], c2[2]) - max(c1[0], c2[0]))
    w1 = max(1e-5, c1[2] - c1[0])
    w2 = max(1e-5, c2[2] - c2[0])
    return inter / min(w1, w2)

def match_boxes(pred_boxes, gt_boxes, iou_thresh=0.5):
    """
    Greedy maximum-IoU bipartite matching between predicted and GT boxes.
    Returns:
        matches: list of (pred_idx, gt_idx, iou)
        unmatched_pred: set of pred_idx
        unmatched_gt: set of gt_idx
    """
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
        if iou_val < iou_thresh:
            break
        p_idx = int(idx // len(gt_boxes))
        g_idx = int(idx % len(gt_boxes))
        
        if p_idx not in matched_preds and g_idx not in matched_gts:
            matched_preds.add(p_idx)
            matched_gts.add(g_idx)
            matches.append((p_idx, g_idx, float(iou_val)))
            
    unmatched_preds = set(range(len(pred_boxes))) - matched_preds
    unmatched_gts = set(range(len(gt_boxes))) - matched_gts
    
    return matches, unmatched_preds, unmatched_gts

def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

def bootstrap_mean_ci(values, B=10000, alpha=0.05, seed=42):
    """Nonparametric bootstrap confidence interval for the mean."""
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    boot_means = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(B)]
    return (float(np.percentile(boot_means, 100 * (alpha / 2))),
            float(np.percentile(boot_means, 100 * (1 - alpha / 2))))

def main():
    images_dir = Path("pubtables_test_sample/images")
    ann_dir = Path("pubtables_test_sample/annotations")
    out_csv = Path("tatr_pubtables1m_consistency_report.csv")
    
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
    
    # TATR structure thresholds
    SCORE_THRESH = 0.5
    SIZE = {"shortest_edge": 800, "longest_edge": 1000}
    
    records = []
    
    for idx, img_path in enumerate(image_paths, 1):
        xml_path = ann_dir / (img_path.stem + ".xml")
        if not xml_path.exists():
            print(f"Warning: missing XML for {img_path.name}")
            continue
            
        # 1. Parse Ground Truth XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        gt_rows = []
        gt_cols = []
        gt_spans = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            b = obj.find("bndbox")
            bbox = [float(b.find(k).text) for k in ["xmin", "ymin", "xmax", "ymax"]]
            if name == "table row":
                gt_rows.append(bbox)
            elif name == "table column":
                gt_cols.append(bbox)
            elif name in ("table spanning cell", "table projected row header"):
                gt_spans.append(bbox)
                
        # 2. Run TATR Forward Pass
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
            if lbl_name == "table row":
                raw_rows.append(obj_dict)
            elif lbl_name == "table column":
                raw_cols.append(obj_dict)
                
        # 3. Analyze RAW Structural Consistency
        n_raw_row = len(raw_rows)
        n_raw_col = len(raw_cols)
        
        # Raw row overlaps
        raw_row_iou_overlaps = 0
        raw_row_vert_overlaps = 0
        for i in range(n_raw_row):
            for j in range(i + 1, n_raw_row):
                iou = compute_box_iou(raw_rows[i]["bbox"], raw_rows[j]["bbox"])
                if iou > 0.05:
                    raw_row_iou_overlaps += 1
                v_ratio = vertical_overlap_ratio(raw_rows[i]["bbox"], raw_rows[j]["bbox"])
                if v_ratio > 0.20:
                    raw_row_vert_overlaps += 1
                    
        # Raw col overlaps
        raw_col_iou_overlaps = 0
        raw_col_horiz_overlaps = 0
        for i in range(n_raw_col):
            for j in range(i + 1, n_raw_col):
                iou = compute_box_iou(raw_cols[i]["bbox"], raw_cols[j]["bbox"])
                if iou > 0.05:
                    raw_col_iou_overlaps += 1
                h_ratio = horizontal_overlap_ratio(raw_cols[i]["bbox"], raw_cols[j]["bbox"])
                if h_ratio > 0.20:
                    raw_col_horiz_overlaps += 1
                    
        total_raw_overlaps = raw_row_iou_overlaps + raw_col_iou_overlaps
        
        # 4. Apply Repair Pipeline (No tokens: tokens=[])
        # refine_rows / columns runs NMS
        rep_rows = postprocess.refine_rows(copy.deepcopy(raw_rows), tokens=[], score_threshold=SCORE_THRESH)
        rep_cols = postprocess.refine_columns(copy.deepcopy(raw_cols), tokens=[], score_threshold=SCORE_THRESH)
        
        # Align rows and columns to table extent
        row_rect = Rect()
        for obj in rep_rows:
            row_rect.include_rect(obj["bbox"])
        col_rect = Rect()
        for obj in rep_cols:
            col_rect.include_rect(obj["bbox"])
            
        table_bbox = [col_rect[0], row_rect[1], col_rect[2], row_rect[3]]
        rep_cols = postprocess.align_columns(rep_cols, table_bbox)
        rep_rows = postprocess.align_rows(rep_rows, table_bbox)
        
        n_rep_row = len(rep_rows)
        n_rep_col = len(rep_cols)
        
        # Repaired overlaps
        rep_row_iou_overlaps = 0
        for i in range(n_rep_row):
            for j in range(i + 1, n_rep_row):
                if compute_box_iou(rep_rows[i]["bbox"], rep_rows[j]["bbox"]) > 0.05:
                    rep_row_iou_overlaps += 1
                    
        rep_col_iou_overlaps = 0
        for i in range(n_rep_col):
            for j in range(i + 1, n_rep_col):
                if compute_box_iou(rep_cols[i]["bbox"], rep_cols[j]["bbox"]) > 0.05:
                    rep_col_iou_overlaps += 1
                    
        total_rep_overlaps = rep_row_iou_overlaps + rep_col_iou_overlaps
        
        # 5. Evaluate Structural Accuracy vs Ground Truth
        # A. Raw Rows vs GT Rows
        raw_row_boxes = [r["bbox"] for r in raw_rows]
        matches_raw_row, _, _ = match_boxes(raw_row_boxes, gt_rows, iou_thresh=0.5)
        tp_raw_row = len(matches_raw_row)
        fp_raw_row = n_raw_row - tp_raw_row
        fn_raw_row = len(gt_rows) - tp_raw_row
        prec_raw_row = tp_raw_row / max(1, n_raw_row)
        rec_raw_row = tp_raw_row / max(1, len(gt_rows))
        f1_raw_row = 2 * prec_raw_row * rec_raw_row / max(1e-6, prec_raw_row + rec_raw_row)
        med_iou_raw_row = float(np.median([m[2] for m in matches_raw_row])) if matches_raw_row else 0.0
        
        # B. Repaired Rows vs GT Rows
        rep_row_boxes = [r["bbox"] for r in rep_rows]
        matches_rep_row, _, _ = match_boxes(rep_row_boxes, gt_rows, iou_thresh=0.5)
        tp_rep_row = len(matches_rep_row)
        fp_rep_row = n_rep_row - tp_rep_row
        fn_rep_row = len(gt_rows) - tp_rep_row
        prec_rep_row = tp_rep_row / max(1, n_rep_row)
        rec_rep_row = tp_rep_row / max(1, len(gt_rows))
        f1_rep_row = 2 * prec_rep_row * rec_rep_row / max(1e-6, prec_rep_row + rec_rep_row)
        med_iou_rep_row = float(np.median([m[2] for m in matches_rep_row])) if matches_rep_row else 0.0
        
        # C. Raw Cols vs GT Cols
        raw_col_boxes = [c["bbox"] for c in raw_cols]
        matches_raw_col, _, _ = match_boxes(raw_col_boxes, gt_cols, iou_thresh=0.5)
        tp_raw_col = len(matches_raw_col)
        fp_raw_col = n_raw_col - tp_raw_col
        fn_raw_col = len(gt_cols) - tp_raw_col
        prec_raw_col = tp_raw_col / max(1, n_raw_col)
        rec_raw_col = tp_raw_col / max(1, len(gt_cols))
        f1_raw_col = 2 * prec_raw_col * rec_raw_col / max(1e-6, prec_raw_col + rec_raw_col)
        med_iou_raw_col = float(np.median([m[2] for m in matches_raw_col])) if matches_raw_col else 0.0
        
        # D. Repaired Cols vs GT Cols
        rep_col_boxes = [c["bbox"] for c in rep_cols]
        matches_rep_col, _, _ = match_boxes(rep_col_boxes, gt_cols, iou_thresh=0.5)
        tp_rep_col = len(matches_rep_col)
        fp_rep_col = n_rep_col - tp_rep_col
        fn_rep_col = len(gt_cols) - tp_rep_col
        prec_rep_col = tp_rep_col / max(1, n_rep_col)
        rec_rep_col = tp_rep_col / max(1, len(gt_cols))
        f1_rep_col = 2 * prec_rep_col * rec_rep_col / max(1e-6, prec_rep_col + rec_rep_col)
        med_iou_rep_col = float(np.median([m[2] for m in matches_rep_col])) if matches_rep_col else 0.0
        
        # Over-pruning checks
        row_pruned_tp = max(0, tp_raw_row - tp_rep_row)
        col_pruned_tp = max(0, tp_raw_col - tp_rep_col)
        
        records.append({
            "table_id": img_path.stem,
            "gt_rows": len(gt_rows),
            "gt_cols": len(gt_cols),
            "raw_rows": n_raw_row,
            "raw_cols": n_raw_col,
            "rep_rows": n_rep_row,
            "rep_cols": n_rep_col,
            "raw_row_iou_overlaps": raw_row_iou_overlaps,
            "raw_row_vert_overlaps": raw_row_vert_overlaps,
            "raw_col_iou_overlaps": raw_col_iou_overlaps,
            "raw_col_horiz_overlaps": raw_col_horiz_overlaps,
            "total_raw_overlaps": total_raw_overlaps,
            "rep_row_iou_overlaps": rep_row_iou_overlaps,
            "rep_col_iou_overlaps": rep_col_iou_overlaps,
            "total_rep_overlaps": total_rep_overlaps,
            "tp_raw_row": tp_raw_row,
            "tp_rep_row": tp_rep_row,
            "tp_raw_col": tp_raw_col,
            "tp_rep_col": tp_rep_col,
            "f1_raw_row": f1_raw_row,
            "f1_rep_row": f1_rep_row,
            "med_iou_raw_row": med_iou_raw_row,
            "med_iou_rep_row": med_iou_rep_row,
            "f1_raw_col": f1_raw_col,
            "f1_rep_col": f1_rep_col,
            "med_iou_raw_col": med_iou_raw_col,
            "med_iou_rep_col": med_iou_rep_col,
            "row_pruned_tp": row_pruned_tp,
            "col_pruned_tp": col_pruned_tp,
        })
        
        if idx % 20 == 0 or idx == len(image_paths):
            print(f"Processed {idx}/{len(image_paths)} tables...")
            
    # Write full CSV report
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved CSV report to {out_csv}")
    
    # 6. Aggregate Statistics and Statistical Rigor
    N = len(records)
    raw_overlap_tables = sum(1 for r in records if r["total_raw_overlaps"] > 0)
    raw_overlap_rate = raw_overlap_tables / N
    raw_ci = wilson_ci(raw_overlap_tables, N)
    
    rep_overlap_tables = sum(1 for r in records if r["total_rep_overlaps"] > 0)
    rep_overlap_rate = rep_overlap_tables / N
    rep_ci = wilson_ci(rep_overlap_tables, N)
    
    # Catastrophic collapse analysis (IQR fence on total raw overlaps)
    all_raw_ov = [r["total_raw_overlaps"] for r in records]
    sorted_ov = sorted(all_raw_ov)
    q1 = statistics.median(sorted_ov[: N // 2])
    q3 = statistics.median(sorted_ov[(N + 1) // 2 :])
    iqr = q3 - q1
    iqr_fence = q3 + 3.0 * iqr if iqr > 0 else float("inf")
    
    catastrophic_tables = [r for r in records if r["total_raw_overlaps"] > iqr_fence]
    cat_count = len(catastrophic_tables)
    cat_rate = cat_count / N
    cat_ci = wilson_ci(cat_count, N)
    
    # Ordinary tables (at or below IQR fence)
    ordinary_tables = [r for r in records if r["total_raw_overlaps"] <= iqr_fence]
    ord_ov = [r["total_raw_overlaps"] for r in ordinary_tables]
    mean_ord_ov = np.mean(ord_ov) if ord_ov else 0.0
    med_ord_ov = np.median(ord_ov) if ord_ov else 0.0
    
    # Micro and Macro F1
    total_tp_raw_row = sum(r["tp_raw_row"] for r in records)
    total_raw_row = sum(r["raw_rows"] for r in records)
    total_gt_row = sum(r["gt_rows"] for r in records)
    micro_p_raw_row = total_tp_raw_row / max(1, total_raw_row)
    micro_r_raw_row = total_tp_raw_row / max(1, total_gt_row)
    micro_f1_raw_row = 2 * micro_p_raw_row * micro_r_raw_row / max(1e-6, micro_p_raw_row + micro_r_raw_row)
    
    total_tp_rep_row = sum(r["tp_rep_row"] for r in records)
    total_rep_row = sum(r["rep_rows"] for r in records)
    micro_p_rep_row = total_tp_rep_row / max(1, total_rep_row)
    micro_r_rep_row = total_tp_rep_row / max(1, total_gt_row)
    micro_f1_rep_row = 2 * micro_p_rep_row * micro_r_rep_row / max(1e-6, micro_p_rep_row + micro_r_rep_row)
    
    total_tp_raw_col = sum(r["tp_raw_col"] for r in records)
    total_raw_col = sum(r["raw_cols"] for r in records)
    total_gt_col = sum(r["gt_cols"] for r in records)
    micro_p_raw_col = total_tp_raw_col / max(1, total_raw_col)
    micro_r_raw_col = total_tp_raw_col / max(1, total_gt_col)
    micro_f1_raw_col = 2 * micro_p_raw_col * micro_r_raw_col / max(1e-6, micro_p_raw_col + micro_r_raw_col)
    
    total_tp_rep_col = sum(r["tp_rep_col"] for r in records)
    total_rep_col = sum(r["rep_cols"] for r in records)
    micro_p_rep_col = total_tp_rep_col / max(1, total_rep_col)
    micro_r_rep_col = total_tp_rep_col / max(1, total_gt_col)
    micro_f1_rep_col = 2 * micro_p_rep_col * micro_r_rep_col / max(1e-6, micro_p_rep_col + micro_r_rep_col)
    
    macro_f1_raw_row = np.mean([r["f1_raw_row"] for r in records])
    macro_f1_rep_row = np.mean([r["f1_rep_row"] for r in records])
    macro_f1_raw_col = np.mean([r["f1_raw_col"] for r in records])
    macro_f1_rep_col = np.mean([r["f1_rep_col"] for r in records])
    
    med_iou_raw_row_agg = np.median([r["med_iou_raw_row"] for r in records if r["med_iou_raw_row"] > 0])
    med_iou_rep_row_agg = np.median([r["med_iou_rep_row"] for r in records if r["med_iou_rep_row"] > 0])
    med_iou_raw_col_agg = np.median([r["med_iou_raw_col"] for r in records if r["med_iou_raw_col"] > 0])
    med_iou_rep_col_agg = np.median([r["med_iou_rep_col"] for r in records if r["med_iou_rep_col"] > 0])
    
    total_row_tp_pruned = sum(r["row_pruned_tp"] for r in records)
    total_col_tp_pruned = sum(r["col_pruned_tp"] for r in records)
    
    print("\n" + "=" * 65)
    print(f"=== TATR (Microsoft Table Transformer) Consistency Summary (N={N}) ===")
    print("=" * 65)
    print(f"Dataset: PubTables-1M (Test split, native domain)")
    print(f"Model:   TATR-v1.1-All (Structure Recognition DETR)")
    print("-" * 65)
    print(f"1. GEOMETRIC OVERLAPS:")
    print(f"   Raw Overlap Rate:        {raw_overlap_tables}/{N} ({raw_overlap_rate*100:.1f}%) [95% CI: {raw_ci[0]*100:.1f}% - {raw_ci[1]*100:.1f}%]")
    print(f"   Post-Repair Overlap:     {rep_overlap_tables}/{N} ({rep_overlap_rate*100:.1f}%) [95% CI: {rep_ci[0]*100:.1f}% - {rep_ci[1]*100:.1f}%]")
    print(f"   Catastrophic Collapse:   {cat_count}/{N} ({cat_rate*100:.1f}%) [95% CI: {cat_ci[0]*100:.1f}% - {cat_ci[1]*100:.1f}%] (fence={iqr_fence:.1f})")
    print(f"   Ordinary Overlaps/Table: Mean={mean_ord_ov:.2f}, Median={med_ord_ov:.1f}")
    print("-" * 65)
    print(f"2. STRUCTURAL ACCURACY (Bipartite IoU >= 0.5 vs Ground Truth):")
    print(f"   ROWS:")
    print(f"     Raw:         Micro F1={micro_f1_raw_row*100:.2f}% (P={micro_p_raw_row*100:.1f}%, R={micro_r_raw_row*100:.1f}%), Macro F1={macro_f1_raw_row*100:.2f}%, MedIoU={med_iou_raw_row_agg:.4f}")
    print(f"     Post-Repair: Micro F1={micro_f1_rep_row*100:.2f}% (P={micro_p_rep_row*100:.1f}%, R={micro_r_rep_row*100:.1f}%), Macro F1={macro_f1_rep_row*100:.2f}%, MedIoU={med_iou_rep_row_agg:.4f}")
    print(f"   COLUMNS:")
    print(f"     Raw:         Micro F1={micro_f1_raw_col*100:.2f}% (P={micro_p_raw_col*100:.1f}%, R={micro_r_raw_col*100:.1f}%), Macro F1={macro_f1_raw_col*100:.2f}%, MedIoU={med_iou_raw_col_agg:.4f}")
    print(f"     Post-Repair: Micro F1={micro_f1_rep_col*100:.2f}% (P={micro_p_rep_col*100:.1f}%, R={micro_r_rep_col*100:.1f}%), Macro F1={macro_f1_rep_col*100:.2f}%, MedIoU={med_iou_rep_col_agg:.4f}")
    print("-" * 65)
    print(f"3. HEURISTIC REPAIR AUDIT & OVER-PRUNING:")
    print(f"   Row True Positives Discarded by Repair: {total_row_tp_pruned}")
    print(f"   Col True Positives Discarded by Repair: {total_col_tp_pruned}")
    print(f"   Circularity Assessment: None (tokens=[] path; row/col extent aligned to table bounds, no token snapping)")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
