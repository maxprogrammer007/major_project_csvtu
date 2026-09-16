import json
import numpy as np
import copy
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
import sys
sys.path.append("d:/RRC/major_project_csvtu")
import tatr_postprocess as postprocess

def compute_box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0: return 0.0
    return inter_area / union_area

def match_boxes(pred_boxes, gt_boxes, iou_thresh=0.5):
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=float)
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_box_iou(pb['bbox'], gb)
    
    matches = []
    matched_preds = set()
    matched_gts = set()
    
    flat_indices = np.argsort(iou_matrix.ravel())[::-1]
    for idx in flat_indices:
        iou_val = iou_matrix.ravel()[idx]
        if iou_val < iou_thresh:
            break
        p_idx = int(idx // len(gt_boxes))
        g_idx = int(idx % len(gt_boxes))
        
        if p_idx not in matched_preds and g_idx not in matched_gts:
            matched_preds.add(p_idx)
            matched_gts.add(g_idx)
            matches.append((p_idx, g_idx, float(iou_val)))
            
    unmatched_preds = set(range(len(pred_boxes))) - matched_preds
    unmatched_gts = set(range(len(gt_boxes))) - matched_gts
    return matches, unmatched_preds, unmatched_gts

def main():
    model_id = "microsoft/table-transformer-structure-recognition-v1.1-all"
    processor = DetrImageProcessor.from_pretrained(model_id)
    model = TableTransformerForObjectDetection.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    img_path = Path("d:/RRC/major_project_csvtu/pubtables_test_sample/images/PMC6127715_table_0.jpg")
    xml_path = Path("d:/RRC/major_project_csvtu/pubtables_test_sample/annotations/PMC6127715_table_0.xml")
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt_rows = []
    for obj in root.findall("object"):
        if obj.find("name").text == "table row":
            b = obj.find("bndbox")
            gt_rows.append([float(b.find(k).text) for k in ["xmin", "ymin", "xmax", "ymax"]])
            
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt", size={"shortest_edge": 800, "longest_edge": 1000})
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    target_sizes = torch.tensor([img.size[::-1]]).to(device)
    results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
    
    raw_rows = []
    for b, s, l in zip(results["boxes"].tolist(), results["scores"].tolist(), results["labels"].tolist()):
        if model.config.id2label[l] == "table row":
            raw_rows.append({"bbox": b, "score": s, "label": l, "orig_idx": len(raw_rows)})
            
    print(f"GT rows: {len(gt_rows)}")
    print(f"Raw rows: {len(raw_rows)}")
    
    raw_matches, raw_unp, raw_ung = match_boxes(raw_rows, gt_rows)
    print(f"Raw TP: {len(raw_matches)}, FP: {len(raw_unp)}, FN: {len(raw_ung)}")
    
    rep_rows = postprocess.refine_rows(copy.deepcopy(raw_rows), tokens=[], score_threshold=0.5)
    print(f"Rep rows: {len(rep_rows)}")
    
    rep_matches, rep_unp, rep_ung = match_boxes(rep_rows, gt_rows)
    print(f"Rep TP: {len(rep_matches)}, FP: {len(rep_unp)}, FN: {len(rep_ung)}")
    
    # Let's see what got pruned
    rep_orig_indices = {r.get('orig_idx', id(r)) for r in rep_rows} 
    # Actually, postprocess.refine_rows might not preserve orig_idx if it creates new dicts. 
    # Let's check postprocess code. It does `aligned_objects.append(obj)`. So it preserves dicts.
    rep_orig_indices = {r['orig_idx'] for r in rep_rows}
    
    pruned_indices = set(range(len(raw_rows))) - rep_orig_indices
    print(f"\nPruned {len(pruned_indices)} rows:")
    
    raw_matched_pred_indices = {p for p, g, i in raw_matches}
    for idx in pruned_indices:
        r = raw_rows[idx]
        status = "TP" if idx in raw_matched_pred_indices else "FP"
        print(f"  Row {idx}: Score={r['score']:.4f}, Status={status}, bbox={r['bbox']}")
        
if __name__ == '__main__':
    main()
