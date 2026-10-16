import torch
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

from transformers import TableTransformerForObjectDetection, DetrImageProcessor
from optimize_tatr_consistency import object2_overlap, solve_mwis, match_boxes

gt_rows = []
xml_path = Path("pubtables_test_sample/annotations/PMC6127715_table_0.xml")
tree = ET.parse(xml_path)
root = tree.getroot()
for obj in root.findall("object"):
    if obj.find("name").text == "table row":
        gt_rows.append([float(obj.find("bndbox").find(k).text) for k in ["xmin", "ymin", "xmax", "ymax"]])

model_id = "microsoft/table-transformer-structure-recognition-v1.1-all"
processor = DetrImageProcessor.from_pretrained(model_id)
model = TableTransformerForObjectDetection.from_pretrained(model_id)
model.eval()

from PIL import Image
img_path = Path("pubtables_test_sample/images/PMC6127715_table_0.jpg")
img = Image.open(img_path).convert("RGB")
inputs = processor(images=img, return_tensors="pt", size={"shortest_edge": 800, "longest_edge": 1000})

with torch.no_grad():
    outputs = model(**inputs)
results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=torch.tensor([img.size[::-1]]))[0]

labels = results["labels"].tolist()
scores = results["scores"].tolist()
boxes = results["boxes"].tolist()

raw_rows = []
for b, s, l in zip(boxes, scores, labels):
    if model.config.id2label[l] == "table row":
        raw_rows.append({"bbox": [float(c) for c in b], "score": float(s)})

med_h = np.median([r['bbox'][3] - r['bbox'][1] for r in raw_rows])
print(f"Median height: {med_h:.2f}")

mod_rows = []
for r in raw_rows:
    h = r['bbox'][3] - r['bbox'][1]
    dist = abs(h - med_h) / med_h
    new_score = max(0.01, 1.0 - dist)
    mod_rows.append({"bbox": r['bbox'], "score": new_score, "raw_score": r["score"], "h": h})

opt_rows, _ = solve_mwis(mod_rows, object2_overlap, threshold=0.5)

raw_bboxes = [r['bbox'] for r in raw_rows]
opt_bboxes = [r['bbox'] for r in opt_rows]

matches_raw, _, _ = match_boxes(raw_bboxes, gt_rows, iou_thresh=0.5)
matches_opt, _, _ = match_boxes(opt_bboxes, gt_rows, iou_thresh=0.5)

raw_tp_bboxes = [raw_bboxes[i] for i, j, _ in matches_raw]
opt_tp_bboxes = [opt_bboxes[i] for i, j, _ in matches_opt]

destroyed_by_opt = [r for r in mod_rows if r['bbox'] in raw_tp_bboxes and r['bbox'] not in opt_tp_bboxes]
print("\nGenuine Rows Destroyed by Height Objective:")
for r in destroyed_by_opt:
    # Find the conflicting box that suppressed it
    conflicts = []
    for o in opt_rows:
        r1, r2 = object2_overlap(r['bbox'], o['bbox'])
        if r1 > 0.5 or r2 > 0.5:
            o_h = o['bbox'][3] - o['bbox'][1]
            conflicts.append((o, o_h))
    
    print(f"  Genuine: Height={r['h']:.2f}, RawScore={r['raw_score']:.4f}, HeightObjScore={r['score']:.4f}")
    for c, c_h in conflicts:
        print(f"    Suppressed by: Height={c_h:.2f}, RawScore={c['raw_score']:.4f}, HeightObjScore={c['score']:.4f}")

