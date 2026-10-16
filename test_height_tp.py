import torch
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

from optimize_tatr_consistency import match_boxes

gt_rows = []
xml_path = Path("pubtables_test_sample/annotations/PMC6127715_table_0.xml")
tree = ET.parse(xml_path)
root = tree.getroot()
for obj in root.findall("object"):
    if obj.find("name").text == "table row":
        gt_rows.append([float(obj.find("bndbox").find(k).text) for k in ["xmin", "ymin", "xmax", "ymax"]])

# I can just re-evaluate match_boxes from the output of the previous script
# Wait, I don't have the bboxes saved from the previous script run.
# I will just run the forward pass again.
from transformers import TableTransformerForObjectDetection, DetrImageProcessor
from PIL import Image
from optimize_tatr_consistency import solve_mwis, object2_overlap

img_path = Path("pubtables_test_sample/images/PMC6127715_table_0.jpg")
model_id = "microsoft/table-transformer-structure-recognition-v1.1-all"
processor = DetrImageProcessor.from_pretrained(model_id)
model = TableTransformerForObjectDetection.from_pretrained(model_id)
model.eval()

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

# Height-based objective
med_h = np.median([r['bbox'][3] - r['bbox'][1] for r in raw_rows])
mod_rows = []
for r in raw_rows:
    h = r['bbox'][3] - r['bbox'][1]
    dist = abs(h - med_h) / med_h
    new_score = max(0.01, 1.0 - dist)
    mod_rows.append({"bbox": r['bbox'], "score": new_score})

opt_rows, _ = solve_mwis(mod_rows, object2_overlap, threshold=0.5)

raw_bboxes = [r['bbox'] for r in raw_rows]
opt_bboxes = [r['bbox'] for r in opt_rows]

matches_raw, _, _ = match_boxes(raw_bboxes, gt_rows, iou_thresh=0.5)
matches_opt, _, _ = match_boxes(opt_bboxes, gt_rows, iou_thresh=0.5)

import tatr_postprocess as postprocess
import copy
heur_rows = postprocess.refine_rows(copy.deepcopy(raw_rows), tokens=[], score_threshold=0.5)
matches_heur, _, _ = match_boxes([r['bbox'] for r in heur_rows], gt_rows, iou_thresh=0.5)

print(f"Raw TP: {len(matches_raw)}")
print(f"Heur TP: {len(matches_heur)}")
print(f"Opt(height) TP: {len(matches_opt)}")
