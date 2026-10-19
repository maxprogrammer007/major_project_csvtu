import sys, os, json
from pathlib import Path
import numpy as np

LORE_DIR = Path("d:/RRC/major_project_csvtu/AdvancedLiterateMachinery/DocumentUnderstanding/LORE-TSR")
os.chdir(str(LORE_DIR))
sys.path.insert(0, str(LORE_DIR / "src/lib"))

from opts import opts
from detectors.detector_factory import detector_factory

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

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

table_data = []
for idx, img_path in enumerate(images):
    json_path = json_dir / (img_path.stem + "_iocr.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt_boxes = [tok['bbox'] for tok in data['pages'][0].get('tokens', [])] if 'pages' in data and len(data['pages']) > 0 else []
    ret = detector.run(opt, str(img_path))
    raw_preds = []
    if 1 in ret.get('results', {}):
        raw_preds = [list(b[:5]) for b in ret['results'][1]]
    table_data.append({"stem": img_path.stem, "gt_boxes": gt_boxes, "preds": raw_preds})

print(f"Loaded and ran {len(table_data)} tables.")

for tau in [0.35, 0.20, 0.01]:
    ambiguous_gt_cells = 0
    tables_with_ambiguity = 0
    total_gt = 0
    total_preds = 0
    for td in table_data:
        gt_boxes = td["gt_boxes"]
        preds = [b for b in td["preds"] if b[4] >= tau]
        total_gt += len(gt_boxes)
        total_preds += len(preds)
        table_ambig = False
        for gb in gt_boxes:
            cands = sum(1 for pb in preds if compute_iou(pb[:4], gb[:4]) >= 0.5)
            if cands >= 2:
                ambiguous_gt_cells += 1
                table_ambig = True
        if table_ambig:
            tables_with_ambiguity += 1
    print(f"tau={tau:.2f}: total_preds={total_preds}, ambiguous_gt_cells={ambiguous_gt_cells}, tables_with_ambiguity={tables_with_ambiguity}/100")
