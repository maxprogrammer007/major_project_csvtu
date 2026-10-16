import torch
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import json
import copy

from transformers import TableTransformerForObjectDetection, DetrImageProcessor
from optimize_tatr_consistency import object2_overlap, solve_mwis, match_boxes
import tatr_postprocess as postprocess

def evaluate_100_tables():
    images_dir = Path("pubtables_test_sample/images")
    annos_dir = Path("pubtables_test_sample/annotations")
    image_paths = sorted(images_dir.glob("*.jpg"))
    
    model_id = "microsoft/table-transformer-structure-recognition-v1.1-all"
    processor = DetrImageProcessor.from_pretrained(model_id)
    model = TableTransformerForObjectDetection.from_pretrained(model_id)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    results = []
    
    for img_path in tqdm(image_paths):
        xml_path = annos_dir / f"{img_path.stem}.xml"
        tree = ET.parse(xml_path)
        root = tree.getroot()
        gt_rows = []
        for obj in root.findall("object"):
            if obj.find("name").text == "table row":
                gt_rows.append([float(obj.find("bndbox").find(k).text) for k in ["xmin", "ymin", "xmax", "ymax"]])
        
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt", size={"shortest_edge": 800, "longest_edge": 1000})
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        res = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=torch.tensor([img.size[::-1]]).to(device))[0]
        labels = res["labels"].tolist()
        scores = res["scores"].tolist()
        boxes = res["boxes"].tolist()
        
        raw_rows = []
        for b, s, l in zip(boxes, scores, labels):
            if model.config.id2label[l] == "table row":
                raw_rows.append({"bbox": [float(c) for c in b], "score": float(s)})
        
        # Heuristic NMS
        heur_rows = postprocess.refine_rows(copy.deepcopy(raw_rows), tokens=[], score_threshold=0.5)
        
        # MWIS with Height Objective
        if len(raw_rows) > 0:
            med_h = np.median([r['bbox'][3] - r['bbox'][1] for r in raw_rows])
            mod_rows = []
            for r in raw_rows:
                h = r['bbox'][3] - r['bbox'][1]
                dist = abs(h - med_h) / med_h
                new_score = max(0.01, 1.0 - dist)
                mod_rows.append({"bbox": r['bbox'], "score": new_score})
            opt_rows, _ = solve_mwis(mod_rows, object2_overlap, threshold=0.5)
        else:
            opt_rows = []
            
        raw_bboxes = [r['bbox'] for r in raw_rows]
        heur_bboxes = [r['bbox'] for r in heur_rows]
        opt_bboxes = [r['bbox'] for r in opt_rows]
        
        raw_tp, _, _ = match_boxes(raw_bboxes, gt_rows, iou_thresh=0.5)
        heur_tp, _, _ = match_boxes(heur_bboxes, gt_rows, iou_thresh=0.5)
        opt_tp, _, _ = match_boxes(opt_bboxes, gt_rows, iou_thresh=0.5)
        
        n_raw_tp = len(raw_tp)
        n_heur_tp = len(heur_tp)
        n_opt_tp = len(opt_tp)
        
        results.append({
            "table_id": img_path.stem,
            "gt_rows": len(gt_rows),
            "raw_tp": n_raw_tp,
            "heur_tp": n_heur_tp,
            "opt_tp": n_opt_tp,
            "heur_destroyed": max(0, n_raw_tp - n_heur_tp),
            "opt_destroyed": max(0, n_raw_tp - n_opt_tp),
        })

    # Summary
    total_gt = sum(r["gt_rows"] for r in results)
    total_raw = sum(r["raw_tp"] for r in results)
    total_heur = sum(r["heur_tp"] for r in results)
    total_opt = sum(r["opt_tp"] for r in results)
    
    heur_destroyed_total = sum(r["heur_destroyed"] for r in results)
    opt_destroyed_total = sum(r["opt_destroyed"] for r in results)
    
    print("\n--- Summary ---")
    print(f"Total GT Rows: {total_gt}")
    print(f"Total Raw TP: {total_raw}")
    print(f"Total Heur NMS TP: {total_heur} (-{heur_destroyed_total} destroyed)")
    print(f"Total Opt Height TP: {total_opt} (-{opt_destroyed_total} destroyed)")
    
    # Win/Loss Accounting
    tables_heur_destroyed = [r for r in results if r["heur_destroyed"] > 0]
    tables_opt_destroyed = [r for r in results if r["opt_destroyed"] > r["heur_destroyed"]]
    
    print(f"\nTables where NMS destroyed TPs: {len(tables_heur_destroyed)}")
    for r in tables_heur_destroyed:
        recovered = r["heur_destroyed"] - r["opt_destroyed"]
        print(f"  {r['table_id']}: NMS destroyed {r['heur_destroyed']}, Opt recovered {recovered} (Net destroyed: {r['opt_destroyed']})")
        
    print(f"\nTables where Opt (Height) performed WORSE than NMS (False Positives introduced/TPs lost): {len(tables_opt_destroyed)}")
    for r in tables_opt_destroyed:
        cost = r["opt_destroyed"] - r["heur_destroyed"]
        print(f"  {r['table_id']}: NMS destroyed {r['heur_destroyed']}, Opt destroyed {r['opt_destroyed']} (Net loss: {cost})")
        
    with open("tatr_height_objective_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    evaluate_100_tables()
