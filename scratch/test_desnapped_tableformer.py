import sys, os, json, cv2
from pathlib import Path
import numpy as np
import pandas as pd

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

def de_snapped_align(table_cells, pdf_cells, matches):
    new_table_cells = []
    table_cell_dict = {cell["cell_id"]: cell for cell in table_cells}
    for pdf_cell_id, match_list in matches.items():
        table_cell_ids = set(int(match["table_cell_id"]) for match in match_list)
        for cell_id in table_cell_ids:
            table_cell = table_cell_dict.get(cell_id)
            if not table_cell: continue
            new_table_cell = table_cell.copy()
            # KEEP ORIGINAL PREDICTED BBOX! DO NOT SNAP TO pdf_cell_bbox!
            new_table_cell["bbox"] = list(table_cell["bbox"])
            if "cell_class" not in new_table_cell:
                new_table_cell["cell_class"] = "2"
            new_table_cells.append(new_table_cell)
    clean_cells = []
    seen = set()
    for c in new_table_cells:
        if c["cell_id"] not in seen:
            seen.add(c["cell_id"])
            clean_cells.append(c)
    return clean_cells

IMG_DIR = Path('pubtab_samples_tokens/images')
JSON_DIR = Path('pubtab_samples_tokens/iocr_jsons')
all_imgs = sorted(list(IMG_DIR.glob('*.png')))

print(f"Running full de-snapped evaluation across all {len(all_imgs)} tables...")
rows = []
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
    
    # 1. Raw
    predictor.enable_post_process = False
    _, details_raw = predictor.predict(page_data, tb, table_image, scale_factor, None, False)
    raw_boxes = [c['bbox'] for c in details_raw.get('table_cells', [])]
    
    # 2. Native snapped post-process
    predictor._post_processor._align_table_cells_to_pdf = orig_align
    predictor.enable_post_process = True
    _, details_snapped = predictor.predict(page_data, tb, table_image, scale_factor, None, True)
    snapped_boxes = [c['bbox'] for c in details_snapped.get('table_cells', [])]
    
    # 3. De-snapped post-process
    predictor._post_processor._align_table_cells_to_pdf = de_snapped_align
    predictor.enable_post_process = True
    _, details_desnapped = predictor.predict(page_data, tb, table_image, scale_factor, None, True)
    desnapped_boxes = [c['bbox'] for c in details_desnapped.get('table_cells', [])]
    
    tp_raw = len(match_boxes(raw_boxes, gt_boxes, 0.5))
    tp_snap = len(match_boxes(snapped_boxes, gt_boxes, 0.5))
    tp_desnap = len(match_boxes(desnapped_boxes, gt_boxes, 0.5))
    
    rows.append({
        'stem': stem,
        'gt': len(gt_boxes),
        'raw_tp': tp_raw,
        'snapped_tp': tp_snap,
        'desnapped_tp': tp_desnap
    })
    if (idx + 1) % 20 == 0:
        print(f"[{idx+1}/{len(all_imgs)}] Raw sum={sum(r['raw_tp'] for r in rows)}, Snapped sum={sum(r['snapped_tp'] for r in rows)}, Desnapped sum={sum(r['desnapped_tp'] for r in rows)}")

df = pd.DataFrame(rows)
df.to_csv("scratch/tableformer_snapping_full_100.csv", index=False)
summary = {
    'total_gt': int(df['gt'].sum()),
    'total_raw_tp': int(df['raw_tp'].sum()),
    'total_snapped_tp': int(df['snapped_tp'].sum()),
    'total_desnapped_tp': int(df['desnapped_tp'].sum()),
    'snapping_inflation': int(df['snapped_tp'].sum() - df['desnapped_tp'].sum()),
    'desnapped_vs_raw': int(df['desnapped_tp'].sum() - df['raw_tp'].sum())
}
with open("scratch/tableformer_snapping_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== FINAL TABLEFORMER SNAPPING RECONCILIATION ===")
print(json.dumps(summary, indent=2))
