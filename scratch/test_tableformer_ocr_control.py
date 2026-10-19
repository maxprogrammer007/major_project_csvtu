import sys, os, json, cv2
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
sys.path.insert(0, 'docling-ibm-models')
from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
from tests.test_tf_predictor import test_config
from huggingface_hub import snapshot_download
from run_full_metric_suite import detect_ocr_tokens

def compute_box_iou(b1, b2):
    x0, y0 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x1, y1 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter_area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0: return 0.0
    return inter_area / union_area

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

download_path = snapshot_download(repo_id='ds4sd/docling-models', revision='v2.1.0', local_files_only=True)
test_config['model']['save_dir'] = os.path.join(download_path, 'model_artifacts/tableformer/fast')
predictor = TFPredictor(test_config, device='cuda', num_threads=2)

orig_align = predictor._post_processor._align_table_cells_to_pdf

IMG_DIR = Path('pubtab_samples_tokens/images')
JSON_DIR = Path('pubtab_samples_tokens/iocr_jsons')
all_imgs = sorted(list(IMG_DIR.glob('*.png')))

print(f"Running TableFormer Decisive Control Experiment (OCR-Text Snapping vs GT Snapping vs De-snapped vs Raw) across {len(all_imgs)} tables...")
rows = []
for idx, img_path in enumerate(all_imgs):
    stem = img_path.stem
    json_path = JSON_DIR / f"{stem}_iocr.json"
    with open(json_path) as fp:
        page_data_gt = json.load(fp)['pages'][0]
    
    img = cv2.imread(str(img_path))
    page_data_gt['image'] = img
    gt_tokens = page_data_gt.get('tokens', [])
    gt_boxes = [t['bbox'] for t in gt_tokens]
    
    page_image_resized, scale_factor = predictor.resize_img(img, height=1024)
    w, h = page_data_gt['width'], page_data_gt['height']
    tb = [0, 0, w * scale_factor, h * scale_factor]
    table_image = page_image_resized[round(tb[1]):round(tb[3]), round(tb[0]):round(tb[2])]
    
    # 1. Raw regressed continuous boxes (no post-processing)
    predictor.enable_post_process = False
    _, details_raw = predictor.predict(page_data_gt, tb, table_image, scale_factor, None, False)
    raw_boxes = [c['bbox'] for c in details_raw.get('table_cells', [])]
    
    # 2. Native pipeline snapped to GT-derived tokens (Circular setup)
    predictor._post_processor._align_table_cells_to_pdf = orig_align
    predictor.enable_post_process = True
    _, details_gt_snap = predictor.predict(page_data_gt, tb, table_image, scale_factor, None, True)
    gt_snap_boxes = [c['bbox'] for c in details_gt_snap.get('table_cells', [])]
    
    # 3. Decisive Control: Native pipeline snapped to realistic OCR-detected word-boxes
    ocr_word_boxes = detect_ocr_tokens(img_path)
    page_data_ocr = {
        'image': img,
        'width': w,
        'height': h,
        'tokens': ocr_word_boxes
    }
    predictor._post_processor._align_table_cells_to_pdf = orig_align
    predictor.enable_post_process = True
    _, details_ocr_snap = predictor.predict(page_data_ocr, tb, table_image, scale_factor, None, True)
    ocr_snap_boxes = [c['bbox'] for c in details_ocr_snap.get('table_cells', [])]
    
    tp_raw = len(match_boxes(raw_boxes, gt_boxes, 0.5))
    tp_gt_snap = len(match_boxes(gt_snap_boxes, gt_boxes, 0.5))
    tp_ocr_snap = len(match_boxes(ocr_snap_boxes, gt_boxes, 0.5))
    
    rows.append({
        'stem': stem,
        'gt': len(gt_boxes),
        'raw_tp': tp_raw,
        'gt_snap_tp': tp_gt_snap,
        'ocr_snap_tp': tp_ocr_snap
    })
    
    if (idx + 1) % 20 == 0:
        r_sum = sum(r['raw_tp'] for r in rows)
        g_sum = sum(r['gt_snap_tp'] for r in rows)
        o_sum = sum(r['ocr_snap_tp'] for r in rows)
        print(f"[{idx+1:3d}/{len(all_imgs)}] Raw={r_sum:4d} | GT-Snap={g_sum:4d} | OCR-Snap={o_sum:4d}")

df = pd.DataFrame(rows)
df.to_csv("scratch/tableformer_ocr_control_results.csv", index=False)

summary = {
    'total_gt': int(df['gt'].sum()),
    'total_raw_tp': int(df['raw_tp'].sum()),
    'total_gt_snap_tp': int(df['gt_snap_tp'].sum()),
    'total_ocr_snap_tp': int(df['ocr_snap_tp'].sum()),
    'gt_snap_gain_over_raw': int(df['gt_snap_tp'].sum() - df['raw_tp'].sum()),
    'ocr_snap_vs_raw': int(df['ocr_snap_tp'].sum() - df['raw_tp'].sum()),
    'ocr_snap_vs_gt_snap': int(df['ocr_snap_tp'].sum() - df['gt_snap_tp'].sum())
}

with open("scratch/tableformer_ocr_control_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("DECISIVE CONTROL EXPERIMENT RESULTS:")
print(json.dumps(summary, indent=2))
print("=" * 60)
