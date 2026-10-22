"""
Zero-Shot Cross-Domain Omission Profiling & Baseline Failure Benchmark
Evaluates Table Transformer (TATR) zero-shot on PubTables-v2 (Smock et al., Dec 2025).

Measures:
1. Base Detector Recall and Precision under Zero-Shot Distribution Shift.
2. Absolute Omission Rate (AOR).
3. Taxonomy of Omissions:
   - Filtered Omissions (covered by tau=0.05 pre-filter candidate pool).
   - Proposal Omissions (completely absent from candidate pool; requiring image fallback).
4. Singh & Middleton (IJDAR 2025) Geometric Completion Baseline:
   - Recovery recall vs False Insertion explosion.
   - Span Violation Rate (mutilation of spanning structures).
5. Threshold Relaxation Baseline (tau=0.05):
   - Competing false candidate proliferation.

Saves outputs strictly in: Major project/
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
from baseline_singh_middleton_2025 import (
    singh_middleton_reconstruction,
    compute_box_iou
)

def match_cells(pred_boxes, gt_boxes, iou_thresh=0.5):
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
    cells = postprocess.table_structure_to_cells(structure, table_spans=[], table_bbox=[0, 0, 3000, 3000])[0]
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

def run_zero_shot_profiling(num_tables=None):
    data_dir = SCRIPT_DIR / "pubtables_v2_test_sample"
    images_dir = data_dir / "images"
    ann_dir = data_dir / "annotations"
    
    image_paths = sorted(images_dir.glob("*.jpg"))
    if num_tables:
        image_paths = image_paths[:num_tables]
        
    print(f"Running Zero-Shot Omission Profiling on {len(image_paths)} PubTables-v2 tables...")
    
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
    total_base_tp = 0
    total_base_fp = 0
    total_omissions = 0
    total_filtered_omissions = 0
    total_proposal_omissions = 0
    
    # Geometric baseline tracking
    total_geom_inferred = 0
    total_geom_valid_recoveries = 0
    total_geom_false_insertions = 0
    total_geom_span_violations = 0
    total_geom_tp = 0
    total_geom_fp = 0
    
    # Threshold relaxation tracking
    total_relax_tp = 0
    total_relax_fp = 0
    
    per_table_results = []
    
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
        res_all = processor.post_process_object_detection(outputs, threshold=0.01, target_sizes=target_sizes)[0]
        
        all_labels = res_all["labels"].tolist()
        all_scores = res_all["scores"].tolist()
        all_boxes = res_all["boxes"].tolist()
        
        # 1. Base detector at operational threshold tau=0.50
        struct_op = extract_tatr_structure(all_boxes, all_scores, all_labels, model.config.id2label, TAU_OP)
        if len(struct_op['rows']) > 0 and len(struct_op['columns']) > 0:
            cells_op = postprocess.table_structure_to_cells(struct_op, table_spans=[], table_bbox=[0, 0, img.size[0], img.size[1]])[0]
            pred_boxes_op = [c['bbox'] for c in cells_op]
        else:
            cells_op = []
            pred_boxes_op = []
            
        # 2. Generous candidate pool at tau=0.05
        struct_gen = extract_tatr_structure(all_boxes, all_scores, all_labels, model.config.id2label, TAU_GENEROUS)
        if len(struct_gen['rows']) > 0 and len(struct_gen['columns']) > 0:
            cells_gen = postprocess.table_structure_to_cells(struct_gen, table_spans=[], table_bbox=[0, 0, img.size[0], img.size[1]])[0]
            pred_boxes_gen = [c['bbox'] for c in cells_gen]
        else:
            pred_boxes_gen = []
            
        # Base matching
        matches_op, un_preds_op, un_gts_op = match_cells(pred_boxes_op, gt_cells, iou_thresh=0.5)
        n_tp = len(matches_op)
        n_fp = len(un_preds_op)
        n_omiss = len(un_gts_op)
        
        # Classify omissions: Filtered vs. Proposal
        n_filt = 0
        n_prop = 0
        for g_idx in un_gts_op:
            gt_b = gt_cells[g_idx]
            max_iou = max([compute_box_iou(gt_b, gb) for gb in pred_boxes_gen], default=0.0)
            if max_iou >= 0.50:
                n_filt += 1
            else:
                n_prop += 1
                
        # Singh & Middleton Geometric Baseline on this table
        inferred, full_geom = singh_middleton_reconstruction(
            detected_rows=struct_op['rows'],
            detected_cols=struct_op['columns'],
            detected_cells=cells_op,
            span_cells=gt_struct['supercells'],
            iou_thresh=0.3
        )
        
        un_gt_boxes = [gt_cells[g] for g in un_gts_op]
        inf_boxes = [c['bbox'] for c in inferred]
        inf_matches, inf_un_preds, _ = match_cells(inf_boxes, un_gt_boxes, iou_thresh=0.5)
        
        n_geom_valid = len(inf_matches)
        n_geom_false = len(inf_un_preds)
        n_geom_span_viol = sum(1 for c in inferred if c.get('span_violation', False))
        
        geom_boxes = [c['bbox'] for c in full_geom]
        geom_matches, geom_un_preds, _ = match_cells(geom_boxes, gt_cells, iou_thresh=0.5)
        
        # Threshold relaxation matching
        relax_matches, relax_un_preds, _ = match_cells(pred_boxes_gen, gt_cells, iou_thresh=0.5)
        
        total_gt_cells += len(gt_cells)
        total_base_tp += n_tp
        total_base_fp += n_fp
        total_omissions += n_omiss
        total_filtered_omissions += n_filt
        total_proposal_omissions += n_prop
        
        total_geom_inferred += len(inferred)
        total_geom_valid_recoveries += n_geom_valid
        total_geom_false_insertions += n_geom_false
        total_geom_span_violations += n_geom_span_viol
        total_geom_tp += len(geom_matches)
        total_geom_fp += len(geom_un_preds)
        
        total_relax_tp += len(relax_matches)
        total_relax_fp += len(relax_un_preds)
        
        per_table_results.append({
            'table_id': img_path.stem,
            'gt_cells': len(gt_cells),
            'base_tp': n_tp,
            'base_fp': n_fp,
            'omissions': n_omiss,
            'filtered_omissions': n_filt,
            'proposal_omissions': n_prop,
            'geom_inferred': len(inferred),
            'geom_valid': n_geom_valid,
            'geom_span_violations': n_geom_span_viol
        })
        
        if idx % 10 == 0 or idx == len(image_paths):
            print(f"Processed {idx}/{len(image_paths)} tables: GT={total_gt_cells}, Base TP={total_base_tp}, Omissions={total_omissions} (Filtered={total_filtered_omissions}, Proposal={total_proposal_omissions})")
            
    base_recall = (total_base_tp / total_gt_cells * 100) if total_gt_cells > 0 else 0.0
    base_prec = (total_base_tp / (total_base_tp + total_base_fp) * 100) if (total_base_tp + total_base_fp) > 0 else 0.0
    aor = (total_omissions / total_gt_cells * 100) if total_gt_cells > 0 else 0.0
    
    filt_pct = (total_filtered_omissions / total_omissions * 100) if total_omissions > 0 else 0.0
    prop_pct = (total_proposal_omissions / total_omissions * 100) if total_omissions > 0 else 0.0
    
    geom_recall = (total_geom_tp / total_gt_cells * 100) if total_gt_cells > 0 else 0.0
    geom_prec = (total_geom_tp / (total_geom_tp + total_geom_fp) * 100) if (total_geom_tp + total_geom_fp) > 0 else 0.0
    geom_svr = (total_geom_span_violations / total_geom_inferred * 100) if total_geom_inferred > 0 else 0.0
    geom_ip = (total_geom_valid_recoveries / total_geom_inferred * 100) if total_geom_inferred > 0 else 0.0
    
    relax_recall = (total_relax_tp / total_gt_cells * 100) if total_gt_cells > 0 else 0.0
    relax_prec = (total_relax_tp / (total_relax_tp + total_relax_fp) * 100) if (total_relax_tp + total_relax_fp) > 0 else 0.0
    
    summary = {
        'benchmark_dataset': 'PubTables-v2 (Zero-Shot Cross-Domain)',
        'total_tables': len(per_table_results),
        'total_gt_cells': total_gt_cells,
        'base_detector': {
            'tp': total_base_tp,
            'fp': total_base_fp,
            'recall_pct': base_recall,
            'precision_pct': base_prec
        },
        'omission_analysis': {
            'total_omissions': total_omissions,
            'absolute_omission_rate_pct': aor,
            'filtered_omissions': total_filtered_omissions,
            'filtered_omissions_pct': filt_pct,
            'proposal_omissions': total_proposal_omissions,
            'proposal_omissions_pct': prop_pct
        },
        'singh_middleton_baseline': {
            'total_inferred': total_geom_inferred,
            'valid_recoveries': total_geom_valid_recoveries,
            'false_insertions': total_geom_false_insertions,
            'span_violations': total_geom_span_violations,
            'span_violation_rate_pct': geom_svr,
            'inferred_precision_pct': geom_ip,
            'reconstructed_recall_pct': geom_recall,
            'reconstructed_precision_pct': geom_prec,
            'precision_penalty_pts': base_prec - geom_prec
        },
        'threshold_relaxation_baseline': {
            'tau': 0.05,
            'reconstructed_recall_pct': relax_recall,
            'reconstructed_precision_pct': relax_prec,
            'precision_penalty_pts': base_prec - relax_prec
        }
    }
    
    summary_out = SCRIPT_DIR / "zero_shot_omission_summary.json"
    per_table_out = SCRIPT_DIR / "zero_shot_omission_per_table.csv"
    
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)
        
    with open(per_table_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            'table_id', 'gt_cells', 'base_tp', 'base_fp', 'omissions',
            'filtered_omissions', 'proposal_omissions', 'geom_inferred',
            'geom_valid', 'geom_span_violations'
        ])
        writer.writeheader()
        writer.writerows(per_table_results)
        
    print("\n" + "="*70)
    print("ZERO-SHOT CROSS-DOMAIN OMISSION PROFILING (PubTables-v2)")
    print("="*70)
    print(f"Total Tables Evaluated:        {summary['total_tables']}")
    print(f"Total Ground Truth Cells:      {total_gt_cells}")
    print(f"Base Detector (Zero-Shot):     Recall={base_recall:.2f}%, Precision={base_prec:.2f}% (TP={total_base_tp}, FP={total_base_fp})")
    print(f"Absolute Omission Rate (AOR):  {aor:.2f}% ({total_omissions} missed cells)")
    print(f"  - Filtered Omissions:        {total_filtered_omissions} ({filt_pct:.2f}% of omissions)")
    print(f"  - Proposal Omissions:        {total_proposal_omissions} ({prop_pct:.2f}% of omissions)")
    print("-" * 70)
    print("NAIVE BASELINE FAILURE SIGNATURES ON CROSS-DOMAIN:")
    print(f"1. Singh & Middleton Geometric Baseline:")
    print(f"   - Inferred Candidates:      {total_geom_inferred}")
    print(f"   - Valid Recoveries:         {total_geom_valid_recoveries} (Precision: {geom_ip:.2f}%)")
    print(f"   - Span Violations:          {total_geom_span_violations} ({geom_svr:.2f}% SVR)")
    print(f"   - Net Precision Cost:       -{base_prec - geom_prec:.2f}% pts ({base_prec:.2f}% -> {geom_prec:.2f}%)")
    print(f"2. Threshold Relaxation Baseline (tau=0.05):")
    print(f"   - Recall:                   {relax_recall:.2f}%")
    print(f"   - Net Precision Cost:       -{base_prec - relax_prec:.2f}% pts ({base_prec:.2f}% -> {relax_prec:.2f}%)")
    print("="*70)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_zero_shot_profiling(n)
