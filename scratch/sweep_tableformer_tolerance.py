"""
Sweep TableFormer Conflict Tolerance to evaluate how much of the 190-cell gap is recovered.
Caches inference results to disk so we only run neural forward passes once.
"""
import sys, os, json, csv, cv2
from pathlib import Path
import numpy as np
import scipy.optimize

sys.path.insert(0, 'docling-ibm-models')
from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
from tests.test_tf_predictor import test_config
from huggingface_hub import snapshot_download

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
    m50, _, _ = match_boxes(pred_boxes, gt_boxes, iou_thresh=0.5)
    tp50 = len(m50)
    prec50 = tp50 / n_pred if n_pred > 0 else 0.0
    rec50 = tp50 / n_gt if n_gt > 0 else 0.0
    f1_50 = 2 * prec50 * rec50 / (prec50 + rec50) if (prec50 + rec50) > 0 else 0.0
    return {'tp': tp50, 'f1': f1_50, 'count': n_pred}

def solve_mwis(raw_boxes, tokens, conflict_fn):
    n = len(raw_boxes)
    if n == 0: return []
    
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
        return raw_boxes
        
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
        return raw_boxes
        
    selected = np.round(res.x).astype(bool)
    return [raw_boxes[i] for i in range(n) if selected[i]]

def get_conflict_fn(criterion, param):
    if criterion == 'strict_inter':
        # intersection > 0
        def fn(b1, b2):
            if b1 == b2: return True
            return compute_intersection_area(b1, b2) > 0.0
        return fn
    elif criterion == 'inter_over_min_area':
        # inter / min(area1, area2) > param
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
        # iou > param
        def fn(b1, b2):
            if b1 == b2: return True
            return compute_box_iou(b1, b2) > param
        return fn
    else:
        raise ValueError(criterion)

def load_or_generate_cache():
    if CACHE_FILE.exists():
        print(f"Loading predictions from cache: {CACHE_FILE}")
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    print("Running TableFormer inference to generate cache...")
    download_path = snapshot_download(repo_id='ds4sd/docling-models', revision='v2.1.0')
    test_config['model']['save_dir'] = os.path.join(download_path, 'model_artifacts/tableformer/fast')
    predictor = TFPredictor(test_config, device='cuda', num_threads=2)
    
    IMG_DIR = Path('pubtab_samples_tokens/images')
    JSON_DIR = Path('pubtab_samples_tokens/iocr_jsons')
    all_imgs = sorted(list(IMG_DIR.glob('*.png')))
    
    cached_data = []
    for idx, img_path in enumerate(all_imgs):
        stem = img_path.stem
        json_path = JSON_DIR / f"{stem}_iocr.json"
        with open(json_path) as fp:
            page_data = json.load(fp)['pages'][0]
        page_data['image'] = cv2.imread(str(img_path))
        tokens = page_data.get('tokens', [])
        gt_boxes = [t['bbox'] for t in tokens]
        
        page_image_resized, scale_factor = predictor.resize_img(page_data['image'], height=1024)
        w, h = page_data['width'], page_data['height']
        tb = [0, 0, w * scale_factor, h * scale_factor]
        table_image = page_image_resized[round(tb[1]):round(tb[3]), round(tb[0]):round(tb[2])]
        
        predictor.enable_post_process = False
        _, details_raw = predictor.predict(page_data, tb, table_image, scale_factor, None, False)
        raw_boxes = [c['bbox'] for c in details_raw.get('table_cells', [])]
        
        predictor.enable_post_process = True
        _, details_post = predictor.predict(page_data, tb, table_image, scale_factor, None, True)
        post_boxes = [c['bbox'] for c in details_post.get('table_cells', [])]
        
        cached_data.append({
            'stem': stem,
            'gt_boxes': gt_boxes,
            'raw_boxes': raw_boxes,
            'post_boxes': post_boxes,
            'tokens': tokens,
            'w': w,
            'h': h
        })
        if (idx + 1) % 20 == 0 or idx == len(all_imgs) - 1:
            print(f"Cached {idx+1}/{len(all_imgs)} tables...")
            
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cached_data, f)
    print("Cache saved.")
    return cached_data

def main():
    tables = load_or_generate_cache()
    
    total_gt = sum(len(t['gt_boxes']) for t in tables)
    raw_tps = sum(evaluate_predictions(t['raw_boxes'], t['gt_boxes'])['tp'] for t in tables)
    post_tps = sum(evaluate_predictions(t['post_boxes'], t['gt_boxes'])['tp'] for t in tables)
    
    print("\n" + "="*80)
    print(f"BASELINES (100 Tables, Total Ground Truth Cells = {total_gt})")
    print(f"  Raw Regressed Boxes TP:            {raw_tps:5d}  (100.0% of Raw Baseline)")
    print(f"  Heuristic Snapping (Post) TP:       {post_tps:5d}  (+{post_tps - raw_tps} vs Raw, due to token snapping circularity)")
    print("="*80)
    
    # Sweep configurations
    configs = [
        ('Strict Planar (intersection > 0)', 'strict_inter', 0.0),
        ('Overlap / min_area > 0.01 (1% bleed tolerance)', 'inter_over_min_area', 0.01),
        ('Overlap / min_area > 0.05 (5% bleed tolerance)', 'inter_over_min_area', 0.05),
        ('Overlap / min_area > 0.10 (10% bleed tolerance)', 'inter_over_min_area', 0.10),
        ('Overlap / min_area > 0.20 (20% bleed tolerance)', 'inter_over_min_area', 0.20),
        ('Overlap / min_area > 0.50 (TATR parity, 50% overlap)', 'inter_over_min_area', 0.50),
        ('IoU > 0.05', 'iou', 0.05),
        ('IoU > 0.10', 'iou', 0.10),
        ('IoU > 0.30', 'iou', 0.30),
        ('IoU > 0.50 (Standard NMS IoU)', 'iou', 0.50),
    ]
    
    sweep_results = []
    
    print(f"{'Configuration':<45} | {'MWIS TP':<8} | {'vs Raw':<10} | {'Gap vs Strict':<16} | {'Gap Recovered (%)':<18}")
    print("-" * 105)
    
    strict_tp = None
    for name, crit, param in configs:
        conflict_fn = get_conflict_fn(crit, param)
        mwis_tp_total = 0
        for t in tables:
            mwis_boxes = solve_mwis(t['raw_boxes'], t['tokens'], conflict_fn)
            ev = evaluate_predictions(mwis_boxes, t['gt_boxes'])
            mwis_tp_total += ev['tp']
            
        diff_raw = mwis_tp_total - raw_tps
        diff_raw_str = f"{diff_raw:+d}"
        
        if strict_tp is None:
            strict_tp = mwis_tp_total
            gap_strict = 0
            pct_recovered_str = "0.0% (Baseline)"
        else:
            gap_strict = mwis_tp_total - strict_tp
            strict_gap_to_raw = raw_tps - strict_tp # 190
            pct_recovered = (gap_strict / strict_gap_to_raw) * 100.0 if strict_gap_to_raw > 0 else 0.0
            pct_recovered_str = f"{pct_recovered:5.1f}% ({gap_strict:+d} cells)"
            
        print(f"{name:<45} | {mwis_tp_total:<8d} | {diff_raw_str:<10} | {gap_strict:+16d} | {pct_recovered_str:<18}")
        
        sweep_results.append({
            'name': name,
            'criterion': crit,
            'param': param,
            'mwis_tp': mwis_tp_total,
            'vs_raw': diff_raw,
            'recovered_vs_strict': mwis_tp_total - (strict_tp or 0)
        })
        
    out_summary = {
        'total_gt': total_gt,
        'raw_tp': raw_tps,
        'post_tp': post_tps,
        'sweep': sweep_results
    }
    with open('scratch/tableformer_tolerance_sweep_results.json', 'w', encoding='utf-8') as f:
        json.dump(out_summary, f, indent=2)
    print("\nSweep results written to scratch/tableformer_tolerance_sweep_results.json")

if __name__ == '__main__':
    main()
