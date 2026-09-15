"""
Sanity check _639579's before/after overlap boxes.
Inspect the actual overlapping pair after post-processing to rule out
a boundary-touching false positive in _boxes_overlap (which uses strict
inequalities: B1[0] >= B2[2] etc., so touching edges do NOT count as overlapping).
"""
import sys, os, json, cv2
sys.path.insert(0, 'docling-ibm-models')
from pathlib import Path
from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
from tests.test_tf_predictor import test_config
from huggingface_hub import snapshot_download
from measure_raw_consistency import _boxes_overlap

download_path = snapshot_download(repo_id='ds4sd/docling-models', revision='v2.1.0')
test_config['model']['save_dir'] = os.path.join(download_path, 'model_artifacts/tableformer/fast')
predictor = TFPredictor(test_config, device='cuda', num_threads=2)

img_path  = Path('pubtab_samples_tokens/images/pubtabnet_tok_639579.png')
json_path = Path('pubtab_samples_tokens/iocr_jsons/pubtabnet_tok_639579_iocr.json')

with open(json_path) as fp:
    iocr_page = json.load(fp)['pages'][0]
iocr_page['image'] = cv2.imread(str(img_path))
page_img, scale = predictor.resize_img(iocr_page['image'], height=1024)
w, h = iocr_page['width'], iocr_page['height']
tb = [0, 0, w * scale, h * scale]
t_img = page_img[round(tb[1]):round(tb[3]), round(tb[0]):round(tb[2])]

# Raw
predictor.enable_post_process = False
_, d_raw = predictor.predict(iocr_page, tb, t_img, scale, None, False)
raw_cells = d_raw.get('table_cells', [])

# Post
predictor.enable_post_process = True
_, d_post = predictor.predict(iocr_page, tb, t_img, scale, None, True)
post_cells = d_post.get('table_cells', [])

print(f"_639579: raw_cells={len(raw_cells)}, post_cells={len(post_cells)}")
print()

# Find all overlapping pairs in post-repair
print("Post-repair overlapping pairs:")
found = 0
for i in range(len(post_cells)):
    for j in range(i+1, len(post_cells)):
        if _boxes_overlap(post_cells[i], post_cells[j]):
            b1 = post_cells[i]['bbox']
            b2 = post_cells[j]['bbox']
            # Compute overlap area
            x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
            overlap_w = x2 - x1
            overlap_h = y2 - y1
            area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
            area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
            print(f"  Pair ({i},{j}):")
            print(f"    cell[{i}] id={post_cells[i].get('cell_id')} bbox={[round(x,2) for x in b1]} area={area1:.1f}")
            print(f"    cell[{j}] id={post_cells[j].get('cell_id')} bbox={[round(x,2) for x in b2]} area={area2:.1f}")
            print(f"    Overlap: w={overlap_w:.2f}, h={overlap_h:.2f}, area={overlap_w*overlap_h:.2f}")
            print(f"    Overlap/area1={overlap_w*overlap_h/area1*100:.1f}%, Overlap/area2={overlap_w*overlap_h/area2*100:.1f}%")
            found += 1

if found == 0:
    print("  (none found — overlap count was 0 on this run)")
    print("  Note: predict() is stochastic if any randomness in model; result may vary.")
