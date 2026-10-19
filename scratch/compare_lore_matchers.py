"""
Compare LORE Matchers: Verbatim official pairTab vs. Canonical Bipartite Matching
Addresses Catch 1:
Demonstrate exact relationship between official LORE pairTab and canonical bipartite matcher
across the 100 benchmark tables.
"""

import sys, os, json, time
from pathlib import Path
import numpy as np

# Change cwd to LORE-TSR directory so model checkpoints resolve
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
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

# Matcher 1: Verbatim official pairTab matching logic
def match_official_pairtab(pred_boxes, gt_boxes, iou_thresh=0.5):
    """
    Verbatim logic from eval_utils.py:
    for tunit in gt_list:
        for sunit in pred_list:
            if compute_IOU >= 0.5: match; break
    """
    tp = 0
    matched_preds = set()
    matched_gt = set()
    for g_idx, gb in enumerate(gt_boxes):
        for p_idx, pb in enumerate(pred_boxes):
            if compute_iou(pb[:4], gb[:4]) >= iou_thresh:
                tp += 1
                matched_preds.add(p_idx)
                matched_gt.add(g_idx)
                break
    fp = len(pred_boxes) - len(matched_preds)
    fn = len(gt_boxes) - tp
    return tp, fp, fn, matched_gt

# Matcher 2: sweep_lore_thresholds.py greedy matcher (per-pred sorted by score)
def match_boxes_greedy_score(pred_boxes, gt_boxes, iou_thresh=0.5):
    matched_gt = set()
    matched_pred = set()
    for i, pb in enumerate(pred_boxes):
        best_iou = 0.0
        best_j = -1
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = compute_iou(pb[:4], gb[:4])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thresh:
            matched_gt.add(best_j)
            matched_pred.add(i)

    tp = len(matched_pred)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn, matched_gt

# Matcher 3: Global bipartite maximum-IoU greedy matching (canonical across project)
def match_global_bipartite(pred_boxes, gt_boxes, iou_thresh=0.5):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0, len(pred_boxes), len(gt_boxes), set()
        
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=float)
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(pb[:4], gb[:4])
            
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
            
    tp = len(matched_preds)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn, matched_gts

def main():
    print("=" * 80)
    print("LORE MATCHER COMPARISON: OFFICIAL PAIRTAB VS CANONICAL BIPARTITE")
    print("=" * 80)

    sys.argv = [
        'smoke_test.py',
        'ctdet_small',
        '--dataset', 'table_small',
        '--arch', 'dla_34',
        '--dataset_name', 'PTN',
        '--tsfm_layers', '3',
        '--stacking_layers', '3',
        '--wiz_2dpe',
        '--wiz_detect',
        '--wiz_stacking',
        '--wiz_rev',
        '--convert_onnx', '0',
        '--vis_thresh_corner', '0.01',
        '--vis_thresh', '0.01',
        '--scores_thresh', '0.01',
        '--nms',
        '--load_model', 'ckpt_ptn/ckpt_ptn/model_best.pth',
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
    print(f"Running inference across {len(images)} benchmark tables...")

    table_data = []
    total_gt = 0

    for idx, img_path in enumerate(images):
        json_path = json_dir / (img_path.stem + "_iocr.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        gt_boxes = []
        if 'pages' in data and len(data['pages']) > 0:
            for tok in data['pages'][0].get('tokens', []):
                gt_boxes.append(tok['bbox'])

        ret = detector.run(opt, str(img_path))
        raw_preds = []
        post_nms_results = ret.get('results', {})
        if 1 in post_nms_results:
            for box in post_nms_results[1]:
                raw_preds.append([float(x) for x in box[:5]])

        total_gt += len(gt_boxes)
        table_data.append({
            "stem": img_path.stem,
            "gt_boxes": gt_boxes,
            "preds": raw_preds
        })

    print(f"\nInference complete. Total GT cells: {total_gt}")

    # Test conditions: tau = 0.35 (default) and tau = 0.20 (optimal)
    conditions = [0.35, 0.20, 0.01]
    
    for tau in conditions:
        print("\n" + "=" * 80)
        print(f"EVALUATION AT CONFIDENCE THRESHOLD TAU = {tau:.2f}")
        print("=" * 80)

        tp_pairtab = 0
        tp_greedy = 0
        tp_bipartite = 0

        total_preds = 0
        identical_tables = 0
        total_overlap_matches = 0
        total_matched_gt_pairtab = 0

        for td in table_data:
            gt_boxes = td["gt_boxes"]
            filtered = [b for b in td["preds"] if b[4] >= tau]
            filtered.sort(key=lambda b: -b[4])
            total_preds += len(filtered)

            t_pt, fp_pt, fn_pt, gt_pt = match_official_pairtab(filtered, gt_boxes, 0.5)
            t_gr, fp_gr, fn_gr, gt_gr = match_boxes_greedy_score(filtered, gt_boxes, 0.5)
            t_bp, fp_bp, fn_bp, gt_bp = match_global_bipartite(filtered, gt_boxes, 0.5)

            tp_pairtab += t_pt
            tp_greedy += t_gr
            tp_bipartite += t_bp

            total_matched_gt_pairtab += len(gt_pt)
            overlap = len(gt_pt.intersection(gt_bp))
            total_overlap_matches += overlap

            if t_pt == t_bp and gt_pt == gt_bp:
                identical_tables += 1

        print(f"Total Predictions: {total_preds}")
        print(f"  1. Verbatim pairTab (Official):   {tp_pairtab:5d} TPs | Rec: {tp_pairtab/total_gt*100:.2f}% | Prec: {tp_pairtab/total_preds*100:.2f}%")
        print(f"  2. Greedy by Score (Sweep):        {tp_greedy:5d} TPs | Rec: {tp_greedy/total_gt*100:.2f}% | Prec: {tp_greedy/total_preds*100:.2f}%")
        print(f"  3. Canonical Global Bipartite:     {tp_bipartite:5d} TPs | Rec: {tp_bipartite/total_gt*100:.2f}% | Prec: {tp_bipartite/total_preds*100:.2f}%")
        
        jaccard_gt = total_overlap_matches / (len(images) * total_gt) # approx
        print(f"  Discrepancy (pairTab vs Bipartite): {tp_pairtab - tp_bipartite:+d} TPs ({abs(tp_pairtab - tp_bipartite)/total_gt*100:.2f}% of corpus)")
        print(f"  Identical Table Matches:           {identical_tables}/{len(images)} tables ({identical_tables/len(images)*100:.1f}%)")
        print(f"  Cell Assignment Agreement:         {total_overlap_matches}/{tp_pairtab} ({total_overlap_matches/tp_pairtab*100:.2f}%)")

if __name__ == '__main__':
    main()
