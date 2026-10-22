import json
import csv
from pathlib import Path
import numpy as np
import scipy.optimize

CACHE_FILE = Path('scratch/tableformer_100_tables_cache.json')
OUT_CSV = Path('scratch/tableformer_fine_sweep_per_table.csv')

def compute_box_iou(b1, b2):
    x0, y0 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x1, y1 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter_area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def compute_intersection_area(b1, b2):
    x0, y0 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x1, y1 = min(b1[2], b2[2]), min(b1[3], b2[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)

def match_boxes(pred_boxes, gt_boxes, iou_thresh=0.5):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=float)
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_box_iou(pb, gb)
    
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
    return len(matched_preds)

def solve_mwis(raw_boxes, tokens, conflict_pairs):
    n = len(raw_boxes)
    if n == 0: return []
    weights = np.zeros(n)
    for i, b in enumerate(raw_boxes):
        box_w = max(0.0, b[2] - b[0])
        box_h = max(0.0, b[3] - b[1])
        box_area = box_w * box_h
        if box_area <= 0:
            weights[i] = 0.0
            continue
        inter_sum = sum(compute_intersection_area(b, t['bbox']) for t in tokens)
        weights[i] = (inter_sum / box_area) + 0.01

    if not conflict_pairs:
        return raw_boxes

    A_ub = np.zeros((len(conflict_pairs), n))
    b_ub = np.ones(len(conflict_pairs))
    for k, (i, j) in enumerate(conflict_pairs):
        A_ub[k, i] = 1
        A_ub[k, j] = 1

    res = scipy.optimize.linprog(-weights, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), integrality=1, method='highs')
    if not res.success:
        return raw_boxes
    return [raw_boxes[i] for i in range(n) if np.round(res.x[i]) > 0.5]

def main():
    print(f"Loading {CACHE_FILE}...")
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        tables = json.load(f)

    # Precompute pairwise IoU and Overlap/min_area for each table
    print("Precomputing pairwise matrices for 100 tables...")
    table_precomp = []
    raw_tps = []
    for t in tables:
        raw_b = t['raw_boxes']
        n = len(raw_b)
        iou_mat = np.zeros((n, n), dtype=float)
        over_mat = np.zeros((n, n), dtype=float)
        for i in range(n):
            b1 = raw_b[i]
            a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
            for j in range(i + 1, n):
                b2 = raw_b[j]
                a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
                inter = compute_intersection_area(b1, b2)
                if inter > 0:
                    union = a1 + a2 - inter
                    iou_mat[i, j] = inter / union if union > 0 else 0.0
                    min_a = min(a1, a2)
                    over_mat[i, j] = inter / min_a if min_a > 0 else 0.0
        r_tp = match_boxes(raw_b, t['gt_boxes'], iou_thresh=0.5)
        raw_tps.append(r_tp)
        table_precomp.append({
            'stem': t['stem'],
            'raw_boxes': raw_b,
            'gt_boxes': t['gt_boxes'],
            'tokens': t['tokens'],
            'raw_tp': r_tp,
            'iou_mat': iou_mat,
            'over_mat': over_mat,
            'n': n
        })

    iou_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
    overlap_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]

    rows = []
    # 1. IoU sweeps
    for tol in iou_thresholds:
        tol_name = f"iou_{tol:.2f}"
        for t in table_precomp:
            n = t['n']
            conflicts = []
            for i in range(n):
                for j in range(i + 1, n):
                    if t['iou_mat'][i, j] > tol:
                        conflicts.append((i, j))
            m_boxes = solve_mwis(t['raw_boxes'], t['tokens'], conflicts)
            m_tp = match_boxes(m_boxes, t['gt_boxes'], iou_thresh=0.5)
            rows.append({
                'source': 'fine_sweep',
                'criterion': 'iou',
                'tolerance': tol_name,
                'tol_value': tol,
                'stem': t['stem'],
                'raw_tp': t['raw_tp'],
                'mwis_tp': m_tp
            })

    # 2. Overlap/min_area sweeps
    for tol in overlap_thresholds:
        tol_name = f"overlap_{tol:.2f}"
        for t in table_precomp:
            n = t['n']
            conflicts = []
            for i in range(n):
                for j in range(i + 1, n):
                    if t['over_mat'][i, j] > tol:
                        conflicts.append((i, j))
            m_boxes = solve_mwis(t['raw_boxes'], t['tokens'], conflicts)
            m_tp = match_boxes(m_boxes, t['gt_boxes'], iou_thresh=0.5)
            rows.append({
                'source': 'fine_sweep',
                'criterion': 'inter_over_min_area',
                'tolerance': tol_name,
                'tol_value': tol,
                'stem': t['stem'],
                'raw_tp': t['raw_tp'],
                'mwis_tp': m_tp
            })

    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'criterion', 'tolerance', 'tol_value', 'stem', 'raw_tp', 'mwis_tp'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {OUT_CSV}")

if __name__ == '__main__':
    main()
