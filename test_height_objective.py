import torch
from transformers import TableTransformerForObjectDetection, DetrImageProcessor
from PIL import Image
from pathlib import Path
import numpy as np

from optimize_tatr_consistency import object2_overlap, solve_mwis

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
    # new score inversely proportional to distance from median height
    dist = abs(h - med_h) / med_h
    new_score = max(0.01, 1.0 - dist)
    mod_rows.append({"bbox": r['bbox'], "score": new_score})

opt_rows, _ = solve_mwis(mod_rows, object2_overlap, threshold=0.5)

print(f"Raw rows: {len(raw_rows)}")
print(f"Opt rows with height objective: {len(opt_rows)}")
