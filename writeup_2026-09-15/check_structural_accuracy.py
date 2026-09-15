"""
Check structural accuracy (IoU-based matching) between TableFormer predictions
(both raw and post-repair) and PubTabNet ground truth cell bboxes.

Addresses the user's critique:
Cell-count agreement with ground truth is NOT the same as structural correctness.
A predicted table can match the ground truth cell count exactly (e.g. 53=53) while
assigning cells to wrong positions or missing key cells entirely.

This script computes:
  - Bipartite / greedy IoU matching between predicted bboxes and GT bboxes
  - True Positives (TP), False Positives (FP), False Negatives (FN)
  - Precision, Recall, F1 at IoU >= 0.5 and IoU >= 0.75
  - Median IoU of matched cells
  - Both for RAW predictions and POST-REPAIR predictions
"""
import sys, os, json, csv, cv2
from pathlib import Path
import numpy as np

sys.path.insert(0, 'docling-ibm-models')
from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
from tests.test_tf_predictor import test_config
from huggingface_hub import snapshot_download

# Target tables: 5 catastrophic + 4 ordinary comparison tables
TARGET_IDS = [
    # Catastrophic collapse (IQR fence > 24)
    'pubtabnet_tok_629558',  # raw=345, post=12, gt=34
    'pubtabnet_tok_634445',  # raw=152, post=122, gt=136
    'pubtabnet_tok_670433',  # raw=59,  post=53,  gt=53
    'pubtabnet_tok_677385',  # raw=285, post=122, gt=137
    'pubtabnet_tok_729453',  # raw=245, post=185, gt=192
    # Ordinary comparison tables
    'pubtabnet_tok_622581',  # raw=35,  post=33,  gt=33
    'pubtabnet_tok_624025',  # raw=51,  post=50,  gt=50
    'pubtabnet_tok_639579',  # raw=76,  post=61,  gt=61 (introduced new overlap)
    'pubtabnet_tok_722429',  # raw=40,  post=29,  gt=46 (37% under GT)
]

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
    
    # Compute pairwise IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=float)
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_box_iou(pb, gb)
    
    # Greedy matching in descending order of IoU
    matches = []
    matched_preds = set()
    matched_gts = set()
    
    # Flatten and sort candidate pairs
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

def evaluate_predictions(pred_boxes, gt_boxes):
    """Compute precision, recall, F1 at 0.5 and 0.75, plus median matched IoU."""
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    
    # IoU >= 0.5
    m50, u_pred50, u_gt50 = match_boxes(pred_boxes, gt_boxes, iou_thresh=0.5)
    tp50 = len(m50)
    fp50 = len(u_pred50)
    fn50 = len(u_gt50)
    prec50 = tp50 / n_pred if n_pred > 0 else 0.0
    rec50 = tp50 / n_gt if n_gt > 0 else 0.0
    f1_50 = 2 * prec50 * rec50 / (prec50 + rec50) if (prec50 + rec50) > 0 else 0.0
    
    # IoU >= 0.75
    m75, u_pred75, u_gt75 = match_boxes(pred_boxes, gt_boxes, iou_thresh=0.75)
    tp75 = len(m75)
    prec75 = tp75 / n_pred if n_pred > 0 else 0.0
    rec75 = tp75 / n_gt if n_gt > 0 else 0.0
    f1_75 = 2 * prec75 * rec75 / (prec75 + rec75) if (prec75 + rec75) > 0 else 0.0
    
    matched_ious = [iou for _, _, iou in m50]
    med_iou = float(np.median(matched_ious)) if matched_ious else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    
    return {
        'n_pred': n_pred,
        'n_gt': n_gt,
        'tp50': tp50,
        'fp50': fp50,
        'fn50': fn50,
        'prec50': prec50,
        'rec50': rec50,
        'f1_50': f1_50,
        'tp75': tp75,
        'prec75': prec75,
        'rec75': rec75,
        'f1_75': f1_75,
        'med_iou50': med_iou,
        'mean_iou50': mean_iou,
    }

def main():
    download_path = snapshot_download(repo_id='ds4sd/docling-models', revision='v2.1.0')
    test_config['model']['save_dir'] = os.path.join(download_path, 'model_artifacts/tableformer/fast')
    predictor = TFPredictor(test_config, device='cuda', num_threads=2)
    
    IMG_DIR = Path('pubtab_samples_tokens/images')
    JSON_DIR = Path('pubtab_samples_tokens/iocr_jsons')
    
    results = []
    
    print(f"{'='*100}")
    print(f"STRUCTURAL ACCURACY EVALUATION: TABLEFORMER PREDICTIONS VS GROUND TRUTH")
    print(f"{'='*100}")
    
    for stem in TARGET_IDS:
        img_path = IMG_DIR / f"{stem}.png"
        json_path = JSON_DIR / f"{stem}_iocr.json"
        
        if not img_path.exists() or not json_path.exists():
            print(f"Skipping {stem}: file not found")
            continue
            
        with open(json_path) as fp:
            page_data = json.load(fp)['pages'][0]
        page_data['image'] = cv2.imread(str(img_path))
        
        # Ground truth non-empty cell bboxes (from tokens built from GT)
        gt_boxes = [t['bbox'] for t in page_data.get('tokens', [])]
        
        page_image_resized, scale_factor = predictor.resize_img(page_data['image'], height=1024)
        w, h = page_data['width'], page_data['height']
        tb = [0, 0, w * scale_factor, h * scale_factor]
        table_image = page_image_resized[round(tb[1]):round(tb[3]), round(tb[0]):round(tb[2])]
        
        # Call 1: Raw predictions (enable_post_process=False)
        predictor.enable_post_process = False
        _, details_raw = predictor.predict(page_data, tb, table_image, scale_factor, None, False)
        raw_boxes = [c['bbox'] for c in details_raw.get('table_cells', [])]
        
        # Call 2: Post-repair predictions (enable_post_process=True)
        predictor.enable_post_process = True
        _, details_post = predictor.predict(page_data, tb, table_image, scale_factor, None, True)
        post_boxes = [c['bbox'] for c in details_post.get('table_cells', [])]
        
        raw_metrics = evaluate_predictions(raw_boxes, gt_boxes)
        post_metrics = evaluate_predictions(post_boxes, gt_boxes)
        
        imgid = stem.replace('pubtabnet_tok_', '')
        res = {
            'stem': stem,
            'imgid': imgid,
            'category': 'catastrophic' if imgid in {'629558', '634445', '670433', '677385', '729453'} else 'ordinary',
            'gt_count': len(gt_boxes),
            'raw_count': len(raw_boxes),
            'post_count': len(post_boxes),
            'raw_metrics': raw_metrics,
            'post_metrics': post_metrics,
        }
        results.append(res)
        
        print(f"\nTable {stem} (GT={len(gt_boxes)}, Category={res['category']}):")
        print(f"  RAW:  count={len(raw_boxes)} | Prec@0.5={raw_metrics['prec50']:.1%} | Rec@0.5={raw_metrics['rec50']:.1%} | F1@0.5={raw_metrics['f1_50']:.1%} | MedIoU={raw_metrics['med_iou50']:.3f}")
        print(f"  POST: count={len(post_boxes)} | Prec@0.5={post_metrics['prec50']:.1%} | Rec@0.5={post_metrics['rec50']:.1%} | F1@0.5={post_metrics['f1_50']:.1%} | MedIoU={post_metrics['med_iou50']:.3f}")
        print(f"  POST: Prec@0.75={post_metrics['prec75']:.1%}, Rec@0.75={post_metrics['rec75']:.1%}, F1@0.75={post_metrics['f1_75']:.1%}")

    # Output detailed summary table
    print(f"\n{'='*110}")
    print(f"SUMMARY TABLE: STRUCTURAL ACCURACY (IoU >= 0.5)")
    print(f"{'='*110}")
    print(f"{'imgid':>8} {'cat':<12} {'gt':>4} | {'raw':>4} {'P_raw':>6} {'R_raw':>6} {'F1_raw':>7} | {'post':>4} {'P_post':>7} {'R_post':>7} {'F1_post':>8} {'MedIoU':>7}")
    print("-" * 110)
    for r in results:
        rm = r['raw_metrics']
        pm = r['post_metrics']
        print(f"{r['imgid']:>8} {r['category']:<12} {r['gt_count']:>4} | "
              f"{r['raw_count']:>4} {rm['prec50']:>6.1%} {rm['rec50']:>6.1%} {rm['f1_50']:>7.1%} | "
              f"{r['post_count']:>4} {pm['prec50']:>7.1%} {pm['rec50']:>7.1%} {pm['f1_50']:>8.1%} {pm['med_iou50']:>7.3f}")

    # Save to CSV
    csv_rows = []
    for r in results:
        rm = r['raw_metrics']
        pm = r['post_metrics']
        csv_rows.append({
            'imgid': r['imgid'],
            'category': r['category'],
            'gt_count': r['gt_count'],
            'raw_count': r['raw_count'],
            'post_count': r['post_count'],
            'raw_prec50': round(rm['prec50'], 4),
            'raw_rec50': round(rm['rec50'], 4),
            'raw_f1_50': round(rm['f1_50'], 4),
            'raw_med_iou50': round(rm['med_iou50'], 4),
            'post_prec50': round(pm['prec50'], 4),
            'post_rec50': round(pm['rec50'], 4),
            'post_f1_50': round(pm['f1_50'], 4),
            'post_prec75': round(pm['prec75'], 4),
            'post_rec75': round(pm['rec75'], 4),
            'post_f1_75': round(pm['f1_75'], 4),
            'post_med_iou50': round(pm['med_iou50'], 4),
        })
    
    out_csv = 'pubtabnet_structural_accuracy_report.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved report to {out_csv}")

if __name__ == '__main__':
    main()
