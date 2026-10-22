import sys, os, json
from pathlib import Path

LORE_DIR = Path("d:/RRC/major_project_csvtu/AdvancedLiterateMachinery/DocumentUnderstanding/LORE-TSR")
os.chdir(str(LORE_DIR))
sys.path.insert(0, str(LORE_DIR / "src/lib"))

from opts import opts
from detectors.detector_factory import detector_factory

sys.argv = [
    'smoke_test.py', 'ctdet_small', '--dataset', 'table_small', '--arch', 'dla_34',
    '--dataset_name', 'PTN', '--tsfm_layers', '3', '--stacking_layers', '3',
    '--wiz_2dpe', '--wiz_detect', '--wiz_stacking', '--wiz_rev', '--convert_onnx', '0',
    '--vis_thresh_corner', '0.01', '--vis_thresh', '0.01', '--scores_thresh', '0.01',
    '--nms', '--load_model', 'ckpt_ptn/ckpt_ptn/model_best.pth',
    '--load_processor', 'ckpt_ptn/ckpt_ptn/processor_best.pth'
]
opt = opts().init()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
Detector = detector_factory[opt.task]
detector = Detector(opt)

base_dir = Path(r"D:\RRC\major_project_csvtu\pubtab_samples_tokens")
img_dir = base_dir / "images"
json_dir = base_dir / "iocr_jsons"
images = sorted(list(img_dir.glob("*.png")))

out_dict = {}
for idx, img_path in enumerate(images):
    json_path = json_dir / (img_path.stem + "_iocr.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt_boxes = [[float(x) for x in tok['bbox']] for tok in data['pages'][0].get('tokens', [])] if 'pages' in data and len(data['pages']) > 0 else []
    ret = detector.run(opt, str(img_path))
    raw_preds = []
    if 1 in ret.get('results', {}):
        raw_preds = [[float(x) for x in b[:5]] for b in ret['results'][1]]
    out_dict[img_path.stem] = {"pred_boxes": raw_preds, "gt_boxes": gt_boxes}
    if (idx + 1) % 25 == 0:
        print(f"Processed {idx+1}/{len(images)} tables...")

out_file = Path(r"D:\RRC\major_project_csvtu\scratch\lore_100_tables_predictions.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(out_dict, f)
print(f"Dumped {len(out_dict)} tables to {out_file}")
