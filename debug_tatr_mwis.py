import torch
from transformers import TableTransformerForObjectDetection, DetrImageProcessor
from PIL import Image
from pathlib import Path
import json

from optimize_tatr_consistency import object2_overlap, solve_mwis
import tatr_postprocess as postprocess
import copy

img_path = Path("d:/RRC/major_project_csvtu/pubtables_test_sample/images/PMC6127715_table_0.jpg")
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
        raw_rows.append({"bbox": [float(coord) for coord in b], "score": float(s), "label": l})

heur_rows = postprocess.refine_rows(copy.deepcopy(raw_rows), tokens=[], score_threshold=0.5)
opt_rows, _ = solve_mwis(copy.deepcopy(raw_rows), object2_overlap, threshold=0.5)

print(f"Raw rows: {len(raw_rows)}")
print(f"Heur rows: {len(heur_rows)}")
print(f"Opt rows: {len(opt_rows)}")

def print_diff(a_rows, b_rows, name_a, name_b):
    a_bboxes = [tuple(r['bbox']) for r in a_rows]
    b_bboxes = [tuple(r['bbox']) for r in b_rows]
    for r in a_rows:
        if tuple(r['bbox']) not in b_bboxes:
            print(f"In {name_a} but not {name_b}: Score {r['score']:.4f}, bbox {r['bbox']}")
            
print("\n--- Differences between Raw and Heur ---")
print_diff(raw_rows, heur_rows, "Raw", "Heur")

print("\n--- Differences between Raw and Opt ---")
print_diff(raw_rows, opt_rows, "Raw", "Opt")
