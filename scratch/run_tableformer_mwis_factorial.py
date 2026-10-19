"""
Run TableFormer MWIS Factorial Evaluation: GT Tokens vs OCR Word-Boxes
Evaluates MWIS on continuous raw regression predictions across both token sources.
"""

import sys, os, json, csv, time, cv2
from pathlib import Path
import numpy as np
import scipy.optimize

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from run_full_metric_suite import detect_ocr_tokens

CACHE_FILE = Path('scratch/tableformer_100_tables_cache.json')
OUT_CSV = Path('scratch/tableformer_mwis_factorial_results.csv')
OUT_JSON = Path('scratch/tableformer_mwis_factorial_summary.json')

def compute_box_iou(b1, b2):
    x0, y0 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x1, y1 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter_area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0: return 0.0
    return inter_area / union_area

def compute_intersection_area(b1, b2):
    x0, y0 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x1, y1 = min(b1[2], b2[2]), min(b1[3], b2[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)

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

def solve_mwis(raw_boxes, tokens, conflict_fn):
    n = len(raw_boxes)
    if n == 0: return [], 0.0
    
    t0 = time.perf_counter()
    c = np.zeros(n)
    for i, b in enumerate(raw_boxes):
        box_w = max(0.0, b[2] - b[0])
        box_h = max(0.0, b[3] - b[1])
        box_area = box_w * box_h
        if box_area <= 0:
            c[i] = -0.01
            continue
            
        inter_sum = 0.0
        for tb in [t['bbox'] for t in tokens]:
            inter = compute_intersection_area(b, tb)
            inter_sum += inter
            
        tightness = inter_sum / box_area
        c[i] = -(tightness + 0.01) # linprog minimizes
        
    conflicts = []
    for i in range(n):
        for j in range(i + 1, n):
            if conflict_fn(raw_boxes[i], raw_boxes[j]):
                conflicts.append((i, j))
                
    if not conflicts:
        solve_time_ms = (time.perf_counter() - t0) * 1000.0
        return raw_boxes, solve_time_ms
        
    A_ub = np.zeros((len(conflicts), n))
    b_ub = np.ones(len(conflicts))
    for k, (i, j) in enumerate(conflicts):
        A_ub[k, i] = 1
        A_ub[k, j] = 1
        
    res = scipy.optimize.linprog(
        c, A_ub=A_ub, b_ub=b_ub,
        bounds=(0, 1), integrality=1,
        method='highs'
    )
    solve_time_ms = (time.perf_counter() - t0) * 1000.0
    
    if not res.success:
        return raw_boxes, solve_time_ms
        
    selected = np.round(res.x).astype(bool)
    return [raw_boxes[i] for i in range(n) if selected[i]], solve_time_ms

def conflict_iou_50(b1, b2):
    if b1 == b2: return True
    return compute_box_iou(b1, b2) > 0.50

def conflict_min_area_50(b1, b2):
    if b1 == b2: return True
    inter = compute_intersection_area(b1, b2)
    if inter <= 0: return False
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    min_a = min(a1, a2)
    if min_a <= 0: return True
    return (inter / min_a) > 0.50

def main():
    print("=" * 80)
    print("TABLEFORMER MWIS FACTORIAL: GT TOKENS VS OCR WORD-BOXES")
    print("=" * 80)
    
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        tables = json.load(f)
    print(f"Loaded {len(tables)} tables from cache.")
    
    IMG_DIR = Path('pubtab_samples_tokens/images')
    
    results = []
    total_gt = 0
    total_raw_tp = 0
    total_mwis_gt_tp = 0
    total_mwis_ocr_tp = 0
    
    mwis_gt_times = []
    mwis_ocr_times = []
    
    for idx, t in enumerate(tables):
        stem = t['stem']
        gt_boxes = t['gt_boxes']
        raw_boxes = t['raw_boxes']
        gt_tokens = t['tokens']
        
        # Load OCR word-boxes
        img_path = IMG_DIR / f"{stem}.png"
        ocr_tokens = detect_ocr_tokens(img_path)
        
        n_gt = len(gt_boxes)
        total_gt += n_gt
        
        raw_tp = len(match_boxes(raw_boxes, gt_boxes, 0.5))
        total_raw_tp += raw_tp
        
        # MWIS with GT tokens (Standard NMS IoU > 0.5)
        boxes_mwis_gt, time_gt = solve_mwis(raw_boxes, gt_tokens, conflict_iou_50)
        tp_mwis_gt = len(match_boxes(boxes_mwis_gt, gt_boxes, 0.5))
        total_mwis_gt_tp += tp_mwis_gt
        mwis_gt_times.append(time_gt)
        
        # MWIS with OCR word-boxes (Standard NMS IoU > 0.5)
        boxes_mwis_ocr, time_ocr = solve_mwis(raw_boxes, ocr_tokens, conflict_iou_50)
        tp_mwis_ocr = len(match_boxes(boxes_mwis_ocr, gt_boxes, 0.5))
        total_mwis_ocr_tp += tp_mwis_ocr
        mwis_ocr_times.append(time_ocr)
        
        results.append({
            'stem': stem,
            'gt_count': n_gt,
            'raw_count': len(raw_boxes),
            'raw_tp': raw_tp,
            'mwis_gt_count': len(boxes_mwis_gt),
            'mwis_gt_tp': tp_mwis_gt,
            'mwis_gt_time_ms': round(time_gt, 2),
            'mwis_ocr_count': len(boxes_mwis_ocr),
            'mwis_ocr_tp': tp_mwis_ocr,
            'mwis_ocr_time_ms': round(time_ocr, 2),
        })
        
    print("\n" + "=" * 80)
    print("GLOBAL RESULTS (100 Tables, Denominator = 5,646 GT Cells)")
    print(f"  Raw Continuous Regressor:   {total_raw_tp:5d} TPs ({total_raw_tp/total_gt*100:.2f}%)")
    print(f"  MWIS (GT Tokens):           {total_mwis_gt_tp:5d} TPs ({total_mwis_gt_tp/total_gt*100:.2f}%) | Mean Time: {np.mean(mwis_gt_times):.2f} ms")
    print(f"  MWIS (Realistic OCR Words): {total_mwis_ocr_tp:5d} TPs ({total_mwis_ocr_tp/total_gt*100:.2f}%) | Mean Time: {np.mean(mwis_ocr_times):.2f} ms")
    print("=" * 80)
    
    # Save CSV
    keys = list(results[0].keys())
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved CSV: {OUT_CSV}")
    
    summary = {
        'total_tables': len(tables),
        'total_gt': total_gt,
        'raw_tp': total_raw_tp,
        'raw_micro_recall': total_raw_tp / total_gt,
        'mwis_gt_tp': total_mwis_gt_tp,
        'mwis_gt_micro_recall': total_mwis_gt_tp / total_gt,
        'mwis_gt_mean_time_ms': float(np.mean(mwis_gt_times)),
        'mwis_ocr_tp': total_mwis_ocr_tp,
        'mwis_ocr_micro_recall': total_mwis_ocr_tp / total_gt,
        'mwis_ocr_mean_time_ms': float(np.mean(mwis_ocr_times)),
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved JSON: {OUT_JSON}")

if __name__ == '__main__':
    main()
