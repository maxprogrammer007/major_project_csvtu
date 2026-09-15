"""
Check two plausible explanations for the 40% vs 57% overlap gap between the
300-table (ajimeno/PubTabNet, no tokens) and 100-table (apoidea/pubtabnet-html,
with tokens) runs — after table-size was already ruled out.

CHECK A: Does match_cells() return different cell *counts* depending on whether
tokens are present? The bbox-mutation audit checked whether coordinates get written;
it did not check whether token presence changes which cells are constructed at all
(e.g. via intersection-matching against tokens affecting candidate filtering).

CHECK B: Do the two Hugging Face mirrors differ in image resolution?
Different resolution -> different scale_factor -> different translated bboxes ->
different overlap geometry, independent of code paths.
"""
import sys, os, json, cv2
sys.path.insert(0, 'docling-ibm-models')
from pathlib import Path
from PIL import Image
import statistics

# --- CHECK B: image resolution comparison ---
# 300-table run: images in pubtab_samples/images/
# 100-table run: images in pubtab_samples_tokens/images/

DIR_300 = Path('pubtab_samples/images')
DIR_100 = Path('pubtab_samples_tokens/images')

def get_resolutions(img_dir, max_n=50):
    sizes = []
    for p in sorted(img_dir.glob('*.png'))[:max_n]:
        img = cv2.imread(str(p))
        if img is not None:
            h, w = img.shape[:2]
            sizes.append((w, h, w * h))
    return sizes

sizes_300 = get_resolutions(DIR_300)
sizes_100 = get_resolutions(DIR_100)

areas_300 = [s[2] for s in sizes_300]
areas_100 = [s[2] for s in sizes_100]
widths_300 = [s[0] for s in sizes_300]
widths_100 = [s[0] for s in sizes_100]
heights_300 = [s[1] for s in sizes_300]
heights_100 = [s[1] for s in sizes_100]

print('=== CHECK B: Image Resolution Comparison ===')
print(f'300-table mirror (ajimeno/PubTabNet):')
print(f'  Width:  mean={statistics.mean(widths_300):.0f}, median={statistics.median(widths_300):.0f}, '
      f'min={min(widths_300)}, max={max(widths_300)}')
print(f'  Height: mean={statistics.mean(heights_300):.0f}, median={statistics.median(heights_300):.0f}, '
      f'min={min(heights_300)}, max={max(heights_300)}')
print(f'  Area:   mean={statistics.mean(areas_300):.0f}, median={statistics.median(areas_300):.0f}')
print()
print(f'100-table mirror (apoidea/pubtabnet-html):')
print(f'  Width:  mean={statistics.mean(widths_100):.0f}, median={statistics.median(widths_100):.0f}, '
      f'min={min(widths_100)}, max={max(widths_100)}')
print(f'  Height: mean={statistics.mean(heights_100):.0f}, median={statistics.median(heights_100):.0f}, '
      f'min={min(heights_100)}, max={max(heights_100)}')
print(f'  Area:   mean={statistics.mean(areas_100):.0f}, median={statistics.median(areas_100):.0f}')

# Flag resolution differences
print()
if abs(statistics.median(areas_300) - statistics.median(areas_100)) > 1000:
    print('RESULT: Median image areas DIFFER significantly -> resolution confound is PLAUSIBLE')
else:
    print('RESULT: Median image areas similar -> resolution difference is NOT the primary explanation')

# --- CHECK A: Does token presence change cell count? ---
# Run the same table image twice — once with tokens, once with empty tokens —
# and compare the count of cells returned by _build_table_cells / match_cells.
# Use first 5 tables from the 100-table set for a quick check.

print()
print('=== CHECK A: Cell Count vs Token Presence (same image, tokens on/off) ===')
from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
from tests.test_tf_predictor import test_config
from huggingface_hub import snapshot_download
import copy

download_path = snapshot_download(repo_id='ds4sd/docling-models', revision='v2.1.0')
test_config['model']['save_dir'] = os.path.join(download_path, 'model_artifacts/tableformer/fast')
predictor = TFPredictor(test_config, device='cuda', num_threads=2)

JSON_DIR = Path('pubtab_samples_tokens/iocr_jsons')
IMG_DIR  = Path('pubtab_samples_tokens/images')

print(f'{"Table":<40} {"Cells(tokens)":>15} {"Cells(no-tokens)":>17} {"Same?":>8}')
print('-' * 85)

for img_path in sorted(IMG_DIR.glob('*.png'))[:8]:
    json_path = JSON_DIR / f'{img_path.stem}_iocr.json'
    if not json_path.exists():
        continue

    with open(json_path) as fp:
        iocr_page_with = json.load(fp)['pages'][0]
    iocr_page_with['image'] = cv2.imread(str(img_path))

    # Version with tokens (original)
    iocr_page_no = copy.deepcopy(iocr_page_with)
    iocr_page_no['tokens'] = []   # strip tokens

    page_img, scale = predictor.resize_img(iocr_page_with['image'], height=1024)
    w, h = iocr_page_with['width'], iocr_page_with['height']
    tb = [0, 0, w * scale, h * scale]
    t_img = page_img[round(tb[1]):round(tb[3]), round(tb[0]):round(tb[2])]

    predictor.enable_post_process = False

    _, d_with = predictor.predict(iocr_page_with, tb, t_img, scale, None, False)
    _, d_no   = predictor.predict(iocr_page_no,   tb, t_img, scale, None, False)

    n_with = len(d_with.get('table_cells', []))
    n_no   = len(d_no.get('table_cells', []))
    same = 'YES' if n_with == n_no else f'NO  ({n_with - n_no:+d})'
    print(f'{img_path.stem:<40} {n_with:>15} {n_no:>17} {same:>8}')

print()
print('If "Same" is always YES: token presence does NOT change cell count -> '
      'raw geometry is identical regardless of tokens.')
print('If any NO: token presence affects which cells are constructed, '
      'making the two runs not directly comparable even before bbox mutations.')
