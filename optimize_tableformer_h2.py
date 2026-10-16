"""
Evaluate TableFormer H2: Optimization Rescue via Token Containment.

Replaces the heuristic MatchingPostProcessor deduplication with an exact MWIS
solver over the native conflict graph. The objective function is Token Containment:
the area of OCR tokens geometrically enclosed by the predicted cell.
"""
import sys, os, json, csv, cv2
from pathlib import Path
import numpy as np
import scipy.optimize

sys.path.insert(0, 'docling-ibm-models')
from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
from tests.test_tf_predictor import test_config
from huggingface_hub import snapshot_download

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

def _boxes_overlap(B1, B2):
    if B1[0] >= B2[2] or B1[2] <= B2[0] or B1[3] <= B2[1] or B1[1] >= B2[3]:
        return False
    return True

def run_mwis(raw_boxes, tokens, w, h, scale_factor):
    n = len(raw_boxes)
    if n == 0: return []
    
    # Map tokens coordinates to match the table image coordinates 
    # tokens are relative to the original image. TableFormer scales them down.
    scaled_tokens = []
    for t in tokens:
        # TFPredictor internally uses tokens, but let's check measure_raw_consistency.py
        # It passes `tb` which is scaled `table_bbox`. The tokens inside `iocr_page` 
        # are in original page coordinates. TFPredictor `_scale_tokens` scales them.
        tb_box = t['bbox']
        scaled_tokens.append([c * scale_factor for c in tb_box])
        
    # Wait, the TableFormer cells are returned in original unscaled page coordinates!
    # Because `MatchingPostProcessor` scales them back before returning.
    # Let's see: `matching_details['table_cells']` has `bbox` in original page coordinates.
    # So we don't need to scale `t['bbox']`. We can just use it directly!
    
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
        
    # 2. Conflicts (with Epsilon Tolerance)
    # Regression naturally bleeds by fractions of a pixel. 
    # Instead of strict intersection > 0, we use intersection over min_area > 0.1
    # to tolerate micro-overlaps between valid adjacent cells.
    conflicts = []
    for i in range(n):
        for j in range(i + 1, n):
            if raw_boxes[i] == raw_boxes[j]:
                conflicts.append((i, j))
            else:
                inter = compute_intersection_area(raw_boxes[i], raw_boxes[j])
                if inter > 0:
                    area_i = max(0.0, raw_boxes[i][2] - raw_boxes[i][0]) * max(0.0, raw_boxes[i][3] - raw_boxes[i][1])
                    area_j = max(0.0, raw_boxes[j][2] - raw_boxes[j][0]) * max(0.0, raw_boxes[j][3] - raw_boxes[j][1])
                    min_area = min(area_i, area_j)
                    if min_area > 0 and (inter / min_area) > 0.1:
                        conflicts.append((i, j))
                
    if not conflicts:
        return raw_boxes
        
    # Exact MWIS solved via HiGHS MILP
    A_sparse = scipy.sparse.csc_matrix(A_ub)
    constraints = scipy.optimize.LinearConstraint(A_sparse, ub=b_ub)
    integrality = np.ones(n)
    bounds = scipy.optimize.Bounds(0, 1)

    res = scipy.optimize.milp(
        c=c, integrality=integrality,
        bounds=bounds, constraints=constraints
    )
    
    if not res.success:
        print(f"MWIS failed: {res.message}")
        return raw_boxes
        
    selected = np.round(res.x).astype(bool)
    return [raw_boxes[i] for i in range(n) if selected[i]]

def main():
    download_path = snapshot_download(repo_id='ds4sd/docling-models', revision='v2.1.0')
    test_config['model']['save_dir'] = os.path.join(download_path, 'model_artifacts/tableformer/fast')
    predictor = TFPredictor(test_config, device='cuda', num_threads=2)
    
    IMG_DIR = Path('pubtab_samples_tokens/images')
    JSON_DIR = Path('pubtab_samples_tokens/iocr_jsons')
    
    results = []
    
    # Run over all 100 images
    all_imgs = sorted(list(IMG_DIR.glob('*.png')))
    print(f"Evaluating {len(all_imgs)} tables...")
    
    for idx, img_path in enumerate(all_imgs):
        stem = img_path.stem
        json_path = JSON_DIR / f"{stem}_iocr.json"
        
        with open(json_path) as fp:
            page_data = json.load(fp)['pages'][0]
        page_data['image'] = cv2.imread(str(img_path))
        
        tokens = page_data.get('tokens', [])
        # Ground truth boxes are exactly the non-empty tokens. 
        # (Wait, check_structural_accuracy.py used gt_boxes = [t['bbox'] for t in tokens])
        gt_boxes = [t['bbox'] for t in tokens]
        
        page_image_resized, scale_factor = predictor.resize_img(page_data['image'], height=1024)
        w, h = page_data['width'], page_data['height']
        tb = [0, 0, w * scale_factor, h * scale_factor]
        table_image = page_image_resized[round(tb[1]):round(tb[3]), round(tb[0]):round(tb[2])]
        
        # Call 1: Raw predictions
        predictor.enable_post_process = False
        _, details_raw = predictor.predict(page_data, tb, table_image, scale_factor, None, False)
        raw_boxes = [c['bbox'] for c in details_raw.get('table_cells', [])]
        
        # Call 2: Post-repair predictions
        predictor.enable_post_process = True
        _, details_post = predictor.predict(page_data, tb, table_image, scale_factor, None, True)
        post_boxes = [c['bbox'] for c in details_post.get('table_cells', [])]
        
        # Call 3: MWIS
        mwis_boxes = run_mwis(raw_boxes, tokens, w, h, scale_factor)
        
        res = {
            'stem': stem,
            'gt': len(gt_boxes),
            'raw': evaluate_predictions(raw_boxes, gt_boxes),
            'post': evaluate_predictions(post_boxes, gt_boxes),
            'mwis': evaluate_predictions(mwis_boxes, gt_boxes)
        }
        results.append(res)
        
        if True:
            print(f"[{idx+1}/{len(all_imgs)}] {stem}: GT={len(gt_boxes)}, Raw={len(raw_boxes)}, Post={len(post_boxes)}, MWIS={len(mwis_boxes)}", flush=True)
            
    # Save detailed
    csv_rows = []
    for r in results:
        csv_rows.append({
            'stem': r['stem'],
            'gt': r['gt'],
            'raw_count': r['raw']['count'],
            'raw_tp': r['raw']['tp'],
            'post_count': r['post']['count'],
            'post_tp': r['post']['tp'],
            'mwis_count': r['mwis']['count'],
            'mwis_tp': r['mwis']['tp'],
        })
    import pandas as pd
    df = pd.DataFrame(csv_rows)
    df.to_csv('tableformer_h2_metrics.csv', index=False)
    
    print("\n--- CORPUS AGGREGATE ---")
    print(f"Total GT cells: {df['gt'].sum()}")
    print(f"Total RAW TPs:  {df['raw_tp'].sum()}")
    print(f"Total POST TPs: {df['post_tp'].sum()}")
    print(f"Total MWIS TPs: {df['mwis_tp'].sum()}")
    
    destroyed_by_post = df['raw_tp'].sum() - df['post_tp'].sum()
    rescued_by_mwis = df['mwis_tp'].sum() - df['post_tp'].sum()
    print(f"\nHeuristic Destruction (H1): {destroyed_by_post} valid TPs thrown away.")
    print(f"Optimization Rescue (H2):   {rescued_by_mwis} valid TPs recovered.")

if __name__ == '__main__':
    main()
