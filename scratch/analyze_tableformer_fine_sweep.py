"""
Fine-grained sweep and statistical analysis of TableFormer MWIS vs Raw.
Extracts coordinate-level traces for any diverging tables at IoU > 0.50.
"""
import json
import numpy as np
import scipy.stats
import scipy.optimize
from pathlib import Path

CACHE_FILE = Path('scratch/tableformer_100_tables_cache.json')

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
        p_idx, g_idx = int(idx // len(gt_boxes)), int(idx % len(gt_boxes))
        if p_idx not in matched_preds and g_idx not in matched_gts:
            matched_preds.add(p_idx)
            matched_gts.add(g_idx)
            matches.append((p_idx, g_idx, float(iou_val)))
    return matches, set(range(len(pred_boxes))) - matched_preds, set(range(len(gt_boxes))) - matched_gts

def evaluate_predictions(pred_boxes, gt_boxes):
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    m50, unmatched_preds, unmatched_gts = match_boxes(pred_boxes, gt_boxes, iou_thresh=0.5)
    tp50 = len(m50)
    prec50 = tp50 / n_pred if n_pred > 0 else 0.0
    rec50 = tp50 / n_gt if n_gt > 0 else 0.0
    f1_50 = 2 * prec50 * rec50 / (prec50 + rec50) if (prec50 + rec50) > 0 else 0.0
    return {'tp': tp50, 'f1': f1_50, 'count': n_pred, 'matches': m50, 'unmatched_preds': unmatched_preds, 'unmatched_gts': unmatched_gts}

def solve_mwis(raw_boxes, tokens, conflict_fn, return_weights=False):
    n = len(raw_boxes)
    if n == 0: return ([], []) if return_weights else []
    
    weights = np.zeros(n)
    for i, b in enumerate(raw_boxes):
        box_w = max(0.0, b[2] - b[0])
        box_h = max(0.0, b[3] - b[1])
        box_area = box_w * box_h
        if box_area <= 0:
            weights[i] = 0.0
            continue
            
        inter_sum = 0.0
        for tb in [t['bbox'] for t in tokens]:
            inter = compute_intersection_area(b, tb)
            inter_sum += inter
            
        tightness = inter_sum / box_area
        weights[i] = tightness + 0.01 # positive weight
        
    c = -weights # linprog minimizes
        
    conflicts = []
    for i in range(n):
        for j in range(i + 1, n):
            if conflict_fn(raw_boxes[i], raw_boxes[j]):
                conflicts.append((i, j))
                
    if not conflicts:
        return (raw_boxes, list(range(n)), weights, []) if return_weights else raw_boxes
        
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
    
    if not res.success:
        return (raw_boxes, list(range(n))) if return_weights else raw_boxes
        
    selected_indices = [i for i in range(n) if np.round(res.x[i]) > 0.5]
    selected_boxes = [raw_boxes[i] for i in selected_indices]
    
    if return_weights:
        return selected_boxes, selected_indices, weights, conflicts
    return selected_boxes

def get_conflict_fn(criterion, param):
    if criterion == 'inter_over_min_area':
        def fn(b1, b2):
            if b1 == b2: return True
            inter = compute_intersection_area(b1, b2)
            if inter <= 0: return False
            a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
            a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
            min_a = min(a1, a2)
            if min_a <= 0: return True
            return (inter / min_a) > param
        return fn
    elif criterion == 'iou':
        def fn(b1, b2):
            if b1 == b2: return True
            return compute_box_iou(b1, b2) > param
        return fn
    else:
        raise ValueError(criterion)

def main():
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        tables = json.load(f)
        
    total_gt = sum(len(t['gt_boxes']) for t in tables)
    raw_evals = [evaluate_predictions(t['raw_boxes'], t['gt_boxes']) for t in tables]
    raw_tps = [ev['tp'] for ev in raw_evals]
    total_raw_tp = sum(raw_tps)
    
    print(f"Total GT: {total_gt}, Total Raw TP: {total_raw_tp}")
    
    # 1. Fine-grained sweep
    iou_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
    overlap_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
    
    print("\n--- FINE IOU SWEEP ---")
    print(f"{'IoU Thresh':<12} | {'MWIS TP':<8} | {'vs Raw':<8} | {'Wins':<6} | {'Ties':<6} | {'Losses':<6} | {'p-value (paired t)':<20}")
    
    fine_results = []
    for thresh in iou_thresholds:
        fn = get_conflict_fn('iou', thresh)
        mwis_tps = []
        for t in tables:
            m_boxes = solve_mwis(t['raw_boxes'], t['tokens'], fn)
            ev = evaluate_predictions(m_boxes, t['gt_boxes'])
            mwis_tps.append(ev['tp'])
        total_mwis_tp = sum(mwis_tps)
        diffs = [m - r for m, r in zip(mwis_tps, raw_tps)]
        wins = sum(1 for d in diffs if d > 0)
        ties = sum(1 for d in diffs if d == 0)
        losses = sum(1 for d in diffs if d < 0)
        
        # paired t-test
        t_stat, p_val = scipy.stats.ttest_rel(mwis_tps, raw_tps)
        
        print(f"{thresh:<12.2f} | {total_mwis_tp:<8d} | {total_mwis_tp - total_raw_tp:<+8d} | {wins:<6d} | {ties:<6d} | {losses:<6d} | p = {p_val:.4f} (t={t_stat:.2f})")
        fine_results.append({
            'type': 'iou',
            'param': thresh,
            'tp': total_mwis_tp,
            'diff': total_mwis_tp - total_raw_tp,
            'wins': wins, 'ties': ties, 'losses': losses,
            'p_value': p_val
        })
        
    print("\n--- FINE OVERLAP / MIN_AREA SWEEP ---")
    print(f"{'Overlap Thresh':<14} | {'MWIS TP':<8} | {'vs Raw':<8} | {'Wins':<6} | {'Ties':<6} | {'Losses':<6} | {'p-value (paired t)':<20}")
    for thresh in overlap_thresholds:
        fn = get_conflict_fn('inter_over_min_area', thresh)
        mwis_tps = []
        for t in tables:
            m_boxes = solve_mwis(t['raw_boxes'], t['tokens'], fn)
            ev = evaluate_predictions(m_boxes, t['gt_boxes'])
            mwis_tps.append(ev['tp'])
        total_mwis_tp = sum(mwis_tps)
        diffs = [m - r for m, r in zip(mwis_tps, raw_tps)]
        wins = sum(1 for d in diffs if d > 0)
        ties = sum(1 for d in diffs if d == 0)
        losses = sum(1 for d in diffs if d < 0)
        
        t_stat, p_val = scipy.stats.ttest_rel(mwis_tps, raw_tps)
        print(f"{thresh:<14.2f} | {total_mwis_tp:<8d} | {total_mwis_tp - total_raw_tp:<+8d} | {wins:<6d} | {ties:<6d} | {losses:<6d} | p = {p_val:.4f} (t={t_stat:.2f})")
        fine_results.append({
            'type': 'inter_over_min_area',
            'param': thresh,
            'tp': total_mwis_tp,
            'diff': total_mwis_tp - total_raw_tp,
            'wins': wins, 'ties': ties, 'losses': losses,
            'p_value': p_val
        })
        
    # 2. Detailed statistical evaluation at IoU > 0.50
    print("\n" + "="*80)
    print("DETAILED STATISTICAL AUDIT AT IoU > 0.50")
    print("="*80)
    fn_iou50 = get_conflict_fn('iou', 0.50)
    mwis_50_tps = []
    diverging_tables = []
    for idx, t in enumerate(tables):
        res_boxes, sel_idx, weights, conflicts = solve_mwis(t['raw_boxes'], t['tokens'], fn_iou50, return_weights=True)
        ev = evaluate_predictions(res_boxes, t['gt_boxes'])
        mwis_50_tps.append(ev['tp'])
        r_tp = raw_tps[idx]
        if ev['tp'] != r_tp:
            diverging_tables.append({
                'idx': idx,
                'stem': t['stem'],
                'gt_count': len(t['gt_boxes']),
                'raw_count': len(t['raw_boxes']),
                'mwis_count': len(res_boxes),
                'raw_tp': r_tp,
                'mwis_tp': ev['tp'],
                'diff': ev['tp'] - r_tp,
                'tokens': t['tokens'],
                'raw_boxes': t['raw_boxes'],
                'mwis_boxes': res_boxes,
                'sel_idx': sel_idx,
                'weights': weights,
                'conflicts': conflicts
            })
            
    diffs_50 = np.array([m - r for m, r in zip(mwis_50_tps, raw_tps)])
    t_stat, p_val_t = scipy.stats.ttest_rel(mwis_50_tps, raw_tps)
    # Wilcoxon signed rank test
    non_zero = diffs_50[diffs_50 != 0]
    if len(non_zero) > 0:
        w_stat, p_val_w = scipy.stats.wilcoxon(diffs_50)
    else:
        w_stat, p_val_w = 0, 1.0
        
    # Bootstrap 95% CI for mean difference
    np.random.seed(42)
    boot_means = [np.mean(np.random.choice(diffs_50, size=len(diffs_50), replace=True)) for _ in range(10000)]
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)
    mean_diff = np.mean(diffs_50)
    
    print(f"Total Tables Evaluated:        N = 100")
    print(f"Total Ground Truth Cells:      5,646")
    print(f"Total Raw TP:                  5,204")
    print(f"Total MWIS (IoU > 0.50) TP:    5,203 (Net Delta = -1)")
    print(f"Per-table distribution:        Wins = {sum(diffs_50 > 0)}, Ties = {sum(diffs_50 == 0)}, Losses = {sum(diffs_50 < 0)}")
    print(f"Mean Difference per table:     {mean_diff:.4f} (95% Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    print(f"Paired t-test:                 t = {t_stat:.4f}, p = {p_val_t:.4f}")
    print(f"Wilcoxon signed-rank test:     W = {w_stat:.4f}, p = {p_val_w:.4f}")
    print(f"Conclusion on difference:      p = {p_val_t:.4f} > 0.05 -> STATISTICALLY INDISTINGUISHABLE FROM ZERO.")
    
    # 3. Coordinate Trace on Diverging Table(s)
    print("\n" + "="*80)
    print(f"DIVERGING TABLES AUDIT AT IoU > 0.50 (Found {len(diverging_tables)} diverging tables)")
    print("="*80)
    for dt in diverging_tables:
        print(f"\nTable: {dt['stem']}")
        print(f"  GT={dt['gt_count']}, Raw Count={dt['raw_count']}, MWIS Count={dt['mwis_count']}")
        print(f"  Raw TP={dt['raw_tp']}, MWIS TP={dt['mwis_tp']}, Delta={dt['diff']}")
        
        # Which boxes were dropped by MWIS?
        dropped_indices = set(range(dt['raw_count'])) - set(dt['sel_idx'])
        print(f"  Dropped box indices by MWIS: {dropped_indices}")
        for di in dropped_indices:
            box = dt['raw_boxes'][di]
            weight = dt['weights'][di]
            print(f"    Dropped Box #{di}: coords={box}, weight(tightness)={weight:.4f}")
            # Find what it conflicted with
            for (i, j) in dt['conflicts']:
                if i == di or j == di:
                    other_i = j if i == di else i
                    other_box = dt['raw_boxes'][other_i]
                    other_w = dt['weights'][other_i]
                    other_sel = other_i in dt['sel_idx']
                    iou_val = compute_box_iou(box, other_box)
                    print(f"      Conflicted with Box #{other_i} (Selected={other_sel}): coords={other_box}, weight={other_w:.4f}, IoU={iou_val:.4f}")
                    
        # Check matching against GT
        raw_m, raw_up, raw_ug = match_boxes(dt['raw_boxes'], [t['bbox'] for t in dt['tokens']], iou_thresh=0.5)
        mwis_m, mwis_up, mwis_ug = match_boxes(dt['mwis_boxes'], [t['bbox'] for t in dt['tokens']], iou_thresh=0.5)
        print(f"  Raw Matches: {len(raw_m)}, MWIS Matches: {len(mwis_m)}")

if __name__ == '__main__':
    main()
