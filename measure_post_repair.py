"""
Measure post-repair geometric overlap across all 100 token-enabled tables.

For each table, run predict() TWICE:
  1. enable_post_process=False -> raw cells (already measured)
  2. enable_post_process=True  -> post-processed cells (NEW measurement)

Run _analyze_raw_cells on BOTH sets of actual cells to get real overlap counts,
not inferred from log messages. This directly answers:
  - What fraction of tables still have >=1 overlap AFTER MatchingPostProcessor?
  - Did the 5 catastrophic tables actually recover, or did overlaps persist?
"""
import sys, os, json, csv, cv2
sys.path.insert(0, 'docling-ibm-models')
from pathlib import Path
from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
from tests.test_tf_predictor import test_config
from huggingface_hub import snapshot_download

# Reuse the same bbox overlap logic from measure_raw_consistency
from measure_raw_consistency import _analyze_raw_cells

download_path = snapshot_download(repo_id='ds4sd/docling-models', revision='v2.1.0')
test_config['model']['save_dir'] = os.path.join(download_path, 'model_artifacts/tableformer/fast')
predictor = TFPredictor(test_config, device='cuda', num_threads=2)

IMG_DIR = Path('pubtab_samples_tokens/images')
JSON_DIR = Path('pubtab_samples_tokens/iocr_jsons')
OUT_CSV  = 'pubtabnet_pre_post_repair_report.csv'

CATASTROPHIC = {
    'pubtabnet_tok_629558_table0',
    'pubtabnet_tok_634445_table0',
    'pubtabnet_tok_670433_table0',
    'pubtabnet_tok_677385_table0',
    'pubtabnet_tok_729453_table0',
}

records = []

for img_path in sorted(IMG_DIR.glob('*.png')):
    table_id = f'{img_path.stem}_table0'
    json_path = JSON_DIR / f'{img_path.stem}_iocr.json'
    if not json_path.exists():
        continue

    with open(json_path) as fp:
        iocr_page = json.load(fp)['pages'][0]
    iocr_page['image'] = cv2.imread(str(img_path))
    page_image_resized, scale_factor = predictor.resize_img(iocr_page['image'], height=1024)
    w, h = iocr_page['width'], iocr_page['height']
    tb = [0, 0, w * scale_factor, h * scale_factor]
    table_image = page_image_resized[round(tb[1]):round(tb[3]), round(tb[0]):round(tb[2])]

    # --- Call 1: raw (no post-processing) ---
    predictor.enable_post_process = False
    _, details_raw = predictor.predict(iocr_page, tb, table_image, scale_factor, None, False)
    raw_cells = details_raw.get('table_cells', [])
    n_raw_ov, n_raw_dup = _analyze_raw_cells(raw_cells)

    # --- Call 2: post-processed ---
    predictor.enable_post_process = True
    _, details_post = predictor.predict(iocr_page, tb, table_image, scale_factor, None, True)
    post_cells = details_post.get('table_cells', [])
    n_post_ov, n_post_dup = _analyze_raw_cells(post_cells)

    record = {
        'table_id':         table_id,
        'is_catastrophic':  table_id in CATASTROPHIC,
        'raw_cell_count':   len(raw_cells),
        'post_cell_count':  len(post_cells),
        'n_raw_overlaps':   n_raw_ov,
        'n_post_overlaps':  n_post_ov,
        'n_raw_dups':       n_raw_dup,
        'n_post_dups':      n_post_dup,
        'overlap_reduced':  n_raw_ov > 0 and n_post_ov < n_raw_ov,
        'fully_recovered':  n_raw_ov > 0 and n_post_ov == 0,
    }
    records.append(record)

    flag = ' [CATASTROPHIC]' if table_id in CATASTROPHIC else ''
    print(f'{table_id}{flag}: raw_ov={n_raw_ov}, post_ov={n_post_ov}, '
          f'raw_cells={len(raw_cells)}, post_cells={len(post_cells)}')

# --- Write CSV ---
with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)

# --- Summary ---
n = len(records)
cat = [r for r in records if r['is_catastrophic']]
ordinary = [r for r in records if not r['is_catastrophic']]

def pct(k, n): return f'{k}/{n} ({100*k/n:.1f}%)'

print(f'\n{"="*60}')
print(f'Pre vs Post Repair: {n} tables')
print(f'{"="*60}')
print(f'\n--- All tables ---')
print(f'Raw >=1 overlap:     {pct(sum(1 for r in records if r["n_raw_overlaps"] > 0), n)}')
print(f'Post >=1 overlap:    {pct(sum(1 for r in records if r["n_post_overlaps"] > 0), n)}')
print(f'Fully recovered:     {pct(sum(1 for r in records if r["fully_recovered"]), sum(1 for r in records if r["n_raw_overlaps"] > 0))} of tables that had raw overlaps')

print(f'\n--- Catastrophic tables (n={len(cat)}) ---')
for r in sorted(cat, key=lambda x: x['n_raw_overlaps'], reverse=True):
    status = 'FULLY RECOVERED' if r['n_post_overlaps'] == 0 else f'{r["n_post_overlaps"]} overlaps remain'
    print(f'  {r["table_id"]}: raw={r["n_raw_overlaps"]:,}, post={r["n_post_overlaps"]:,} -> {status}')

print(f'\n--- Ordinary tables (n={len(ordinary)}) ---')
ov_ord = [r for r in ordinary if r['n_raw_overlaps'] > 0]
print(f'Had raw overlaps: {pct(len(ov_ord), len(ordinary))}')
still_ov = [r for r in ov_ord if r['n_post_overlaps'] > 0]
print(f'Still have post overlaps: {pct(len(still_ov), len(ov_ord))} of those')
if still_ov:
    print('  Tables with residual overlaps after repair:')
    for r in sorted(still_ov, key=lambda x: x['n_post_overlaps'], reverse=True)[:10]:
        print(f'    {r["table_id"]}: raw={r["n_raw_overlaps"]}, post={r["n_post_overlaps"]}')

print(f'\nReport written to: {OUT_CSV}')
