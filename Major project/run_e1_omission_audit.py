"""
E1: Go/No-Go Experiment for Table Cell Insertion in TSR.
Evaluates Microsoft Table Transformer (TATR-v1.1-All) on PubTables-1M test split.

Quantifies:
1. Total Ground Truth cells across test tables.
2. Detected cells at standard operating threshold (tau = 0.5).
3. Missed cells (Omissions) at tau = 0.5.
4. Classification of missed cells:
   - Filtered Omissions: Recovered by generous candidate pool (tau = 0.05 or lower threshold).
   - Proposal Omissions: Not recovered even in generous candidate pool (absence of query/proposal).
5. Decision rule evaluation: proposal omissions >= 5% -> proceed with full three-way design.
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

import torch
import numpy as np
from PIL import Image
from transformers import TableTransformerForObjectDetection, DetrImageProcessor
from fitz import Rect

import tatr_postprocess as postprocess

def compute_box_iou(b1, b2):
    """Compute IoU between two bboxes [x0, y0, x1, y1]."""
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2])
    y1 = min(b1[3], b2[3])
    
    inter_w = max(0.0, x1 - x0)
    inter_h = max(0.0, y1 - y0)
    inter_area = inter_w * inter_h
    
    area1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def match_cells(pred_boxes, gt_boxes, iou_thresh=0.5):
    """Greedy bipartite matching by IoU."""
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

def extract_tatr_structure(boxes, scores, labels, id2label, score_thresh):
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

def run_e1(num_tables=None):
    images_dir = SCRIPT_DIR / "pubtables_test_sample/images"
    if not images_dir.exists():
        images_dir = WORKSPACE_DIR / "pubtables_test_sample/images"
    ann_dir = SCRIPT_DIR / "pubtables_test_sample/annotations"
    if not images_dir.exists() or not ann_dir.exists():
        ann_dir = WORKSPACE_DIR / "pubtables_test_sample/annotations"
    image_paths = sorted(images_dir.glob("*.jpg"))
    if num_tables:
        image_paths = image_paths[:num_tables]
        
    print(f"Running E1 on {len(image_paths)} tables from {images_dir}...")
    
    model_id = "microsoft/table-transformer-structure-recognition-v1.1-all"
    processor = DetrImageProcessor.from_pretrained(model_id)
    model = TableTransformerForObjectDetection.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    SIZE = {"shortest_edge": 800, "longest_edge": 1000}
    TAU_OP = 0.50
    TAU_GENEROUS = 0.05
    
    total_gt_cells = 0
    total_det_tp = 0
    total_det_fp = 0
    total_omissions = 0
    total_filtered_omissions = 0
    total_proposal_omissions = 0
    
    # Also track row and column omissions directly
    total_gt_rows = 0
    total_gt_cols = 0
    missed_gt_rows = 0
    missed_gt_cols = 0
    filtered_missed_rows = 0
    proposal_missed_rows = 0
    filtered_missed_cols = 0
    proposal_missed_cols = 0
    
    per_table_results = []
    
    for idx, img_path in enumerate(image_paths, 1):
        xml_path = ann_dir / (img_path.stem + ".xml")
        if not xml_path.exists():
            continue
            
        gt_struct, gt_cells = parse_gt_xml(xml_path)
        if len(gt_cells) == 0:
            continue
            
        gt_row_boxes = [r['bbox'] for r in gt_struct['rows']]
        gt_col_boxes = [c['bbox'] for c in gt_struct['columns']]
        
        # Load and run inference
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt", size=SIZE)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        target_sizes = torch.tensor([img.size[::-1]]).to(device)
        # Extract generous predictions down to 0.01 threshold
        res_all = processor.post_process_object_detection(outputs, threshold=0.01, target_sizes=target_sizes)[0]
        
        all_labels = res_all["labels"].tolist()
        all_scores = res_all["scores"].tolist()
        all_boxes = res_all["boxes"].tolist()
        
        # 1. Operational predictions at tau = 0.50
        struct_op = extract_tatr_structure(all_boxes, all_scores, all_labels, model.config.id2label, TAU_OP)
        if len(struct_op['rows']) > 0 and len(struct_op['columns']) > 0:
            cells_op = postprocess.table_structure_to_cells(struct_op, table_spans=[], table_bbox=[0, 0, img.size[0], img.size[1]])[0]
            pred_cell_boxes_op = [c['bbox'] for c in cells_op]
        else:
            pred_cell_boxes_op = []
            
        # 2. Generous predictions at tau = 0.05
        struct_gen = extract_tatr_structure(all_boxes, all_scores, all_labels, model.config.id2label, TAU_GENEROUS)
        if len(struct_gen['rows']) > 0 and len(struct_gen['columns']) > 0:
            cells_gen = postprocess.table_structure_to_cells(struct_gen, table_spans=[], table_bbox=[0, 0, img.size[0], img.size[1]])[0]
            pred_cell_boxes_gen = [c['bbox'] for c in cells_gen]
        else:
            pred_cell_boxes_gen = []
            
        # Match operational cells to GT
        matches_op, unmatched_pred_op, unmatched_gt_op = match_cells(pred_cell_boxes_op, gt_cells, iou_thresh=0.5)
        tp_cells = len(matches_op)
        fp_cells = len(unmatched_pred_op)
        n_omissions = len(unmatched_gt_op)
        
        # Classify omissions: check if covered by generous pool
        n_filtered = 0
        n_proposal = 0
        for g_idx in unmatched_gt_op:
            gt_box = gt_cells[g_idx]
            # Check maximum IoU with any box in generous candidate pool
            max_iou = 0.0
            for gen_box in pred_cell_boxes_gen:
                iou = compute_box_iou(gt_box, gen_box)
                if iou > max_iou:
                    max_iou = iou
            if max_iou >= 0.50:
                n_filtered += 1
            else:
                n_proposal += 1
                
        # Also component-level analysis (rows and cols)
        op_row_boxes = [r['bbox'] for r in struct_op['rows']]
        gen_row_boxes = [r['bbox'] for r in struct_gen['rows']]
        _, _, un_gt_rows = match_cells(op_row_boxes, gt_row_boxes, iou_thresh=0.5)
        for r_idx in un_gt_rows:
            gt_r = gt_row_boxes[r_idx]
            max_r_iou = max([compute_box_iou(gt_r, gb) for gb in gen_row_boxes], default=0.0)
            if max_r_iou >= 0.5:
                filtered_missed_rows += 1
            else:
                proposal_missed_rows += 1
        missed_gt_rows += len(un_gt_rows)
        total_gt_rows += len(gt_row_boxes)
        
        op_col_boxes = [c['bbox'] for c in struct_op['columns']]
        gen_col_boxes = [c['bbox'] for c in struct_gen['columns']]
        _, _, un_gt_cols = match_cells(op_col_boxes, gt_col_boxes, iou_thresh=0.5)
        for c_idx in un_gt_cols:
            gt_c = gt_col_boxes[c_idx]
            max_c_iou = max([compute_box_iou(gt_c, gb) for gb in gen_col_boxes], default=0.0)
            if max_c_iou >= 0.5:
                filtered_missed_cols += 1
            else:
                proposal_missed_cols += 1
        missed_gt_cols += len(un_gt_cols)
        total_gt_cols += len(gt_col_boxes)
        
        total_gt_cells += len(gt_cells)
        total_det_tp += tp_cells
        total_det_fp += fp_cells
        total_omissions += n_omissions
        total_filtered_omissions += n_filtered
        total_proposal_omissions += n_proposal
        
        per_table_results.append({
            'table_id': img_path.stem,
            'gt_cells': len(gt_cells),
            'tp_cells': tp_cells,
            'fp_cells': fp_cells,
            'omissions': n_omissions,
            'filtered_omissions': n_filtered,
            'proposal_omissions': n_proposal
        })
        
        if idx % 10 == 0 or idx == len(image_paths):
            print(f"Processed {idx}/{len(image_paths)} tables: GT={total_gt_cells}, TP={total_det_tp}, Omissions={total_omissions} (Filtered={total_filtered_omissions}, Proposal={total_proposal_omissions})")
            
    omission_rate = (total_omissions / total_gt_cells * 100) if total_gt_cells > 0 else 0.0
    filtered_pct = (total_filtered_omissions / total_omissions * 100) if total_omissions > 0 else 0.0
    proposal_pct = (total_proposal_omissions / total_omissions * 100) if total_omissions > 0 else 0.0
    
    summary = {
        'total_tables': len(per_table_results),
        'total_gt_cells': total_gt_cells,
        'detected_tp_cells': total_det_tp,
        'cell_recall_tau_0_5': (total_det_tp / total_gt_cells * 100) if total_gt_cells > 0 else 0.0,
        'cell_precision_tau_0_5': (total_det_tp / (total_det_tp + total_det_fp) * 100) if (total_det_tp + total_det_fp) > 0 else 0.0,
        'total_omissions': total_omissions,
        'omission_rate_pct': omission_rate,
        'filtered_omissions': total_filtered_omissions,
        'filtered_omissions_pct': filtered_pct,
        'proposal_omissions': total_proposal_omissions,
        'proposal_omissions_pct': proposal_pct,
        'row_omissions': {
            'total_gt_rows': total_gt_rows,
            'missed_rows': missed_gt_rows,
            'filtered_missed_rows': filtered_missed_rows,
            'proposal_missed_rows': proposal_missed_rows
        },
        'col_omissions': {
            'total_gt_cols': total_gt_cols,
            'missed_cols': missed_gt_cols,
            'filtered_missed_cols': filtered_missed_cols,
            'proposal_missed_cols': proposal_missed_cols
        },
        'decision_rule_proposal_ge_5pct': bool(proposal_pct >= 5.0)
    }
    
    summary_file = SCRIPT_DIR / "e1_omission_audit_summary.json"
    per_table_file = SCRIPT_DIR / "e1_omission_audit_per_table.csv"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
        
    with open(per_table_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['table_id', 'gt_cells', 'tp_cells', 'fp_cells', 'omissions', 'filtered_omissions', 'proposal_omissions'])
        writer.writeheader()
        writer.writerows(per_table_results)
        
    print("\n" + "="*60)
    print("E1 GO/NO-GO EXPERIMENT AUDIT RESULTS")
    print("="*60)
    print(f"Total Tables Evaluated:        {summary['total_tables']}")
    print(f"Total Ground Truth Cells:      {total_gt_cells}")
    print(f"Detected Cells at tau=0.5:     {total_det_tp} (Recall: {summary['cell_recall_tau_0_5']:.2f}%)")
    print(f"Total Missed Cells (Omissions):{total_omissions} ({omission_rate:.2f}% of GT)")
    print(f"  - Filtered Omissions:        {total_filtered_omissions} ({filtered_pct:.2f}% of omissions)")
    print(f"  - Proposal Omissions:        {total_proposal_omissions} ({proposal_pct:.2f}% of omissions)")
    print("-" * 60)
    print(f"Row misses: {missed_gt_rows}/{total_gt_rows} (Filtered: {filtered_missed_rows}, Proposal: {proposal_missed_rows})")
    print(f"Col misses: {missed_gt_cols}/{total_gt_cols} (Filtered: {filtered_missed_cols}, Proposal: {proposal_missed_cols})")
    print("-" * 60)
    if proposal_pct >= 5.0:
        print(f"DECISION RULE VERDICT: Proposal omissions = {proposal_pct:.2f}% (>= 5.0%).")
        print(">> PROCEED WITH FULL THREE-WAY DESIGN (insert / no-insert / span-compatible).")
        print(">> Image-grounded fallback is essential to recover the proposal omissions.")
    else:
        print(f"DECISION RULE VERDICT: Filtered omissions dominate ({filtered_pct:.2f}%).")
        print(">> Candidate-set-conditioned insertion is exceptionally well positioned.")
        print(">> Proceed with targeted candidate-recovery head.")
    print("="*60)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_e1(n)
