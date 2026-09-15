"""
Verify: do the 5 catastrophic-collapse tables recover after MatchingPostProcessor runs?
Re-runs each flagged table with enable_post_process=True and measures final overlap count.
"""
import sys, os, json, cv2
sys.path.insert(0, 'docling-ibm-models')
from pathlib import Path
from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
from tests.test_tf_predictor import test_config
from huggingface_hub import snapshot_download

download_path = snapshot_download(repo_id='ds4sd/docling-models', revision='v2.1.0')
test_config['model']['save_dir'] = os.path.join(download_path, 'model_artifacts/tableformer/fast')
predictor = TFPredictor(test_config, device='cuda', num_threads=2)

IMG_DIR = Path('pubtab_samples_tokens/images')
JSON_DIR = Path('pubtab_samples_tokens/iocr_jsons')

CATASTROPHIC = {
    'pubtabnet_tok_629558': {'raw_overlaps': 33766, 'cells': 345},
    'pubtabnet_tok_634445': {'raw_overlaps': 34,    'cells': 152},
    'pubtabnet_tok_670433': {'raw_overlaps': 28,    'cells': 59},
    'pubtabnet_tok_677385': {'raw_overlaps': 45,    'cells': 285},
    'pubtabnet_tok_729453': {'raw_overlaps': 40,    'cells': 245},
}

def count_overlaps(table_cells):
    """Count overlapping pairs in the final (post-repair) cell list."""
    n_overlaps = 0
    n = len(table_cells)
    for i in range(n):
        b1 = table_cells[i].get('bbox', [])
        for j in range(i + 1, n):
            b2 = table_cells[j].get('bbox', [])
            if not b1 or not b2:
                continue
            if b1[0] >= b2[2] or b1[2] <= b2[0] or b1[3] <= b2[1] or b1[1] >= b2[3]:
                continue
            n_overlaps += 1
    return n_overlaps

print('=== Catastrophic Collapse Recovery Check ===')
print(f'{"Table":<35} {"Raw_Ov":>8} {"Cells":>7} {"Post_Ov":>9} {"Recovered?":>12}')
print('-' * 75)

for stem, info in sorted(CATASTROPHIC.items()):
    img_path = IMG_DIR / f'{stem}.png'
    json_path = JSON_DIR / f'{stem}_iocr.json'

    if not img_path.exists() or not json_path.exists():
        print(f'{stem:<35} MISSING FILES')
        continue

    with open(json_path) as fp:
        iocr_page = json.load(fp)['pages'][0]
    iocr_page['image'] = cv2.imread(str(img_path))
    page_image_resized, scale_factor = predictor.resize_img(iocr_page['image'], height=1024)
    w, h = iocr_page['width'], iocr_page['height']
    tb = [0, 0, w * scale_factor, h * scale_factor]
    table_image = page_image_resized[round(tb[1]):round(tb[3]), round(tb[0]):round(tb[2])]

    # Post-processing ENABLED (real measurement)
    predictor.enable_post_process = True
    _, details_post = predictor.predict(iocr_page, tb, table_image, scale_factor, None, True)
    post_cells = details_post.get('table_cells', [])
    post_ov = count_overlaps(post_cells)

    raw_ov = info['raw_overlaps']
    recovered = 'YES' if post_ov == 0 else f'NO ({post_ov} remain)'
    print(f'{stem:<35} {raw_ov:>8,} {info["cells"]:>7} {post_ov:>9,} {recovered:>12}')

print()
print('Note: post_ov counts overlapping pairs in the final cell list AFTER')
print('_move_cells_to_left_pos, _align_table_cells_to_pdf, _find_overlapping.')
