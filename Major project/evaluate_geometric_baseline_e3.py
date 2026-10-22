"""
Evaluate Singh & Middleton (IJDAR 2025) Geometric Completion Baseline (E3)
against TATR Base Detector on PubTables-1M 100 test tables.

Measures:
- Initial Base True Positives, False Positives, Omissions at tau=0.5
- Singh & Middleton Inferred Candidates
- Valid Recoveries (matched to GT omissions with IoU >= 0.5)
- False Insertions (inferred candidates with no GT match)
- Span Violation Rate: inferred candidates that subdivide/fracture GT spanning cells
- Final reconstructed Precision and Recall
"""

import os
import sys
import json
import csv
from pathlib import Path
import xml.etree.ElementTree as ET

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
from PIL import Image
import torch
from transformers import TableTransformerForObjectDetection, DetrImageProcessor
from fitz import Rect

import tatr_postprocess as postprocess
from baseline_singh_middleton_2025 import singh_middleton_reconstruction, compute_box_iou

def match_boxes(pred_boxes, gt_boxes, iou_thresh=0.5):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return [], set(range(len(pred_boxes))), set(range(len(gt_boxes)))
    
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=float)
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_box_iou(pb, gb)
            
    matches = []
    matched_preds = set()
    matched_gts = set()
    
    flat_indices = np.argsort(iou_matrix.ravel())[::-1]
    for idx in flat_indices:
        val = iou_matrix.ravel()[idx]
        if val < iou_thresh:
            break
        p_idx = int(idx // len(gt_boxes))
        g_idx = int(idx % len(gt_boxes))
        
        if p_idx not in matched_preds and g_idx not in matched_gts:
            matched_preds.add(p_idx)
            matched_gts.add(g_idx)
            matches.append((p_idx, g_idx, float(val)))
            
    unmatched_preds = set(range(len(pred_boxes))) - matched_preds
    unmatched_gts = set(range(len(gt_boxes))) - matched_gts
    return matches, unmatched_preds, unmatched_gts

def parse_gt_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows, cols, spans = [], [], []
    for obj in root.findall("object"):
        name = obj.find("name").text
        b = obj.find("bndbox")
        bbox = [float(b.find(k).text) for k in ["xmin", "ymin", "xmax", "ymax"]]
        if name == "table row":
            rows.append({'bbox': bbox, 'header': False})
        elif name == "table column":
            cols.append({'bbox': bbox})
        elif name in ("table spanning cell", "table projected row header"):
            spans.append({'bbox': bbox, 'subheader': (name == "table projected row header")})
            
    rows.sort(key=lambda x: x['bbox'][1])
    cols.sort(key=lambda x: x['bbox'][0])
    structure = {'rows': rows, 'columns': cols, 'supercells': spans}
    cells = postprocess.table_structure_to_cells(structure, table_spans=[], table_bbox=[0, 0, 2000, 2000])[0]
    gt_cell_boxes = [c['bbox'] for c in cells]
    return structure, gt_cell_boxes

def extract_tatr_structure(boxes, scores, labels, id2label, score_thresh=0.5):
    rows, cols, spans = [], [], []
    for b, s, l in zip(boxes, scores, labels):
        if s < score_thresh:
            continue
        lbl_name = id2label[l]
        bbox = [float(c) for c in b]
        if lbl_name == "table row":
            rows.append({'bbox': bbox, 'score': float(s), 'header': False})
        elif lbl_name == "table column":
            cols.append({'bbox': bbox, 'score': float(s)})
        elif lbl_name in ("table spanning cell", "table projected row header"):
            spans.append({'bbox': bbox, 'score': float(s), 'subheader': (lbl_name == "table projected row header")})
            
    rows.sort(key=lambda x: x['bbox'][1])
    cols.sort(key=lambda x: x['bbox'][0])
    return {'rows': rows, 'columns': cols, 'supercells': spans}

def main():
    images_dir = SCRIPT_DIR / "pubtables_test_sample/images"
    if not images_dir.exists():
        images_dir = WORKSPACE_DIR / "pubtables_test_sample/images"
    ann_dir = SCRIPT_DIR / "pubtables_test_sample/annotations"
    if not images_dir.exists() or not ann_dir.exists():
        ann_dir = WORKSPACE_DIR / "pubtables_test_sample/annotations"
    image_paths = sorted(images_dir.glob("*.jpg"))
    print(f"Evaluating Singh & Middleton Baseline on {len(image_paths)} tables from {images_dir}...")
    
    model_id = "microsoft/table-transformer-structure-recognition-v1.1-all"
    processor = DetrImageProcessor.from_pretrained(model_id)
    model = TableTransformerForObjectDetection.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    SIZE = {"shortest_edge": 800, "longest_edge": 1000}
    
    tot_gt_cells = 0
    base_tp = 0
    base_fp = 0
    
    total_inferred = 0
    total_inferred_valid_recoveries = 0
    total_inferred_false_insertions = 0
    total_span_violations = 0
    
    recon_tp = 0
    recon_fp = 0
    
    for idx, img_path in enumerate(image_paths, 1):
        xml_path = ann_dir / (img_path.stem + ".xml")
        if not xml_path.exists():
            continue
            
        gt_struct, gt_cells = parse_gt_xml(xml_path)
        if len(gt_cells) == 0:
            continue
            
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt", size=SIZE)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            
        target_sizes = torch.tensor([img.size[::-1]]).to(device)
        res = processor.post_process_object_detection(outputs, threshold=0.50, target_sizes=target_sizes)[0]
        
        all_labels = res["labels"].tolist()
        all_scores = res["scores"].tolist()
        all_boxes = res["boxes"].tolist()
        
        struct = extract_tatr_structure(all_boxes, all_scores, all_labels, model.config.id2label, score_thresh=0.50)
        if len(struct['rows']) > 0 and len(struct['columns']) > 0:
            det_cells = postprocess.table_structure_to_cells(struct, table_spans=[], table_bbox=[0, 0, img.size[0], img.size[1]])[0]
        else:
            det_cells = []
            
        det_cell_boxes = [c['bbox'] for c in det_cells]
        
        # Base matching
        matches, un_preds, un_gts = match_boxes(det_cell_boxes, gt_cells, iou_thresh=0.5)
        base_tp += len(matches)
        base_fp += len(un_preds)
        tot_gt_cells += len(gt_cells)
        
        # Run Singh & Middleton (IJDAR 2025) geometric completion
        inferred, full_recon = singh_middleton_reconstruction(
            detected_rows=struct['rows'],
            detected_cols=struct['columns'],
            detected_cells=det_cells,
            span_cells=gt_struct['supercells'],
            iou_thresh=0.3
        )
        
        total_inferred += len(inferred)
        
        # Evaluate inferred cells against un-matched GT cells (true omissions)
        un_gt_boxes = [gt_cells[g] for g in un_gts]
        inferred_boxes = [c['bbox'] for c in inferred]
        
        inf_matches, inf_un_preds, _ = match_boxes(inferred_boxes, un_gt_boxes, iou_thresh=0.5)
        total_inferred_valid_recoveries += len(inf_matches)
        total_inferred_false_insertions += len(inf_un_preds)
        
        for c in inferred:
            if c.get('span_violation', False):
                total_span_violations += 1
                
        # Full reconstructed set matching
        full_recon_boxes = [c['bbox'] for c in full_recon]
        recon_matches, recon_un_preds, _ = match_boxes(full_recon_boxes, gt_cells, iou_thresh=0.5)
        recon_tp += len(recon_matches)
        recon_fp += len(recon_un_preds)
        
    base_rec = base_tp / tot_gt_cells * 100
    base_prec = base_tp / (base_tp + base_fp) * 100
    recon_rec = recon_tp / tot_gt_cells * 100
    recon_prec = recon_tp / (recon_tp + recon_fp) * 100
    
    summary = {
        'total_gt_cells': tot_gt_cells,
        'base_detector': {
            'tp': base_tp,
            'fp': base_fp,
            'recall_pct': base_rec,
            'precision_pct': base_prec
        },
        'singh_middleton_baseline': {
            'total_inferred_cells': total_inferred,
            'valid_recoveries': total_inferred_valid_recoveries,
            'false_insertions': total_inferred_false_insertions,
            'span_violations': total_span_violations,
            'span_violation_rate_pct': (total_span_violations / total_inferred * 100) if total_inferred > 0 else 0.0,
            'inferred_precision_pct': (total_inferred_valid_recoveries / total_inferred * 100) if total_inferred > 0 else 0.0,
            'full_reconstructed_tp': recon_tp,
            'full_reconstructed_fp': recon_fp,
            'full_reconstructed_recall_pct': recon_rec,
            'full_reconstructed_precision_pct': recon_prec
        }
    }
    
    out_file = SCRIPT_DIR / "singh_middleton_baseline_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "="*65)
    print("SINGH & MIDDLETON (IJDAR 2025) GEOMETRIC COMPLETION AUDIT")
    print("="*65)
    print(f"Total Ground Truth Cells:       {tot_gt_cells}")
    print(f"Base Detector (TATR tau=0.5):   Recall={base_rec:.2f}%, Precision={base_prec:.2f}% (TP={base_tp}, FP={base_fp})")
    print("-" * 65)
    print(f"Total Inferred Candidates:      {total_inferred}")
    print(f"Valid Cell Recoveries:          {total_inferred_valid_recoveries}")
    print(f"False Insertions (Unmatched):   {total_inferred_false_insertions}")
    print(f"Inferred Precision:             {summary['singh_middleton_baseline']['inferred_precision_pct']:.2f}%")
    print(f"Span Violations (Fractures):    {total_span_violations} ({summary['singh_middleton_baseline']['span_violation_rate_pct']:.2f}% of inferences)")
    print("-" * 65)
    print(f"Full Reconstructed Metrics:     Recall={recon_rec:.2f}%, Precision={recon_prec:.2f}% (TP={recon_tp}, FP={recon_fp})")
    print(f"Net Recall Delta:               +{recon_rec - base_rec:.2f}%")
    print(f"Net Precision Penalty:          -{base_prec - recon_prec:.2f}%")
    print("="*65)

if __name__ == "__main__":
    main()
