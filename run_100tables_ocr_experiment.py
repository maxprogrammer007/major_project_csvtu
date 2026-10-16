"""
100-Table OCR vs Ground-Truth Token Experiment for SPLERGE
===========================================================
Primary research question:
  Does SPLERGE's content-aware post-processing (H1 heuristic and H2 optimizer)
  still work when the content signal comes from an actual OCR system rather
  than privileged/oracle ground-truth annotations?

Experimental Matrix:
  Condition   Grid          Content signal   Purpose
  ---------   ----------    --------------   --------------------------------
  Raw         SplitModel    None             Baseline (no token signal)
  H1-GT       SplitModel    GT tokens        Privileged heuristic repair
  H1-OCR      SplitModel    OCR tokens       Realistic heuristic repair
  H2-GT       SplitModel    GT tokens        Privileged ILP optimizer
  H2-OCR      SplitModel    OCR tokens       Main realistic experiment

Controlled Invariants:
  1. Exactly the same 100 table image crops across all conditions.
  2. Exactly the same SplitModel checkpoint (split_model.pth) producing identical C_raw.
  3. Exactly the same H1 merge pipeline (merge.py) and H2 ILP formulation (milp).
  4. Exactly the same evaluation protocol (evaluate_matching at IoU >= 0.25 vs PubTabNet GT).
  5. ONLY the token source is replaced: T_GT -> T_OCR.
"""

import sys
import json
import time
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp
import scipy.stats

# -- SPLERGE imports -----------------------------------------------------------
SPLERGE_PATH = Path("d:/RRC/major_project_csvtu/AdvancedLiterateMachinery/DocumentUnderstanding/SPLERGE")
sys.path.insert(0, str(SPLERGE_PATH))
from libs.split_model_hf import SplitModel
import libs.GTElement as gc
import libs.merge_utility as mut
import merge as merge_mod

# -- Paths ---------------------------------------------------------------------
IMG_DIR    = Path("d:/RRC/major_project_csvtu/pubtab_samples_tokens/images")
JSON_DIR   = Path("d:/RRC/major_project_csvtu/pubtab_samples_tokens/iocr_jsons")
MODEL_PATH = SPLERGE_PATH / "model_out/split_model.pth"
OUT_CSV    = Path("d:/RRC/major_project_csvtu/ocr_100tables_results.csv")
OUT_JSON   = Path("d:/RRC/major_project_csvtu/ocr_100tables_summary.json")


# ==============================================================================
# 1. OCR TOKEN DETECTOR (Morphological Word-Level Tokenizer)
# ==============================================================================

def detect_ocr_tokens(img_path: Path):
    """
    Realistic word-level OCR token detector.
    
    Binarises the table crop with Otsu's thresholding and applies horizontal
    morphological closing (kernel 7x2) to merge character glyphs into word
    bounding boxes, producing [id, 'word', x0, y0, x1, y1] directly from the image
    without oracle PubTabNet cell geometry.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 2))
    dilated = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    
    img_area = img.shape[0] * img.shape[1]
    tokens = []
    tok_id = 0
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        # Filter noise and whole-image bounding boxes
        if area < 6 or area > img_area * 0.15:
            continue
        if w < 3 or h < 3:
            continue
        tokens.append([tok_id, "word", int(x), int(y), int(x + w), int(y + h)])
        tok_id += 1
    return tokens


# ==============================================================================
# 2. SPLIT-MODEL INFERENCE (Identical C_raw for all conditions)
# ==============================================================================

def get_middle_of_groups(binary_array):
    result = np.zeros_like(binary_array)
    i, n = 0, len(binary_array)
    while i < n:
        if binary_array[i] == 1:
            start = i
            while i < n and binary_array[i] == 1:
                i += 1
            result[(start + i - 1) // 2] = 1
        else:
            i += 1
    return result


def run_split_model(model, transform, img: Image.Image, device):
    """Return (row_seps, col_seps) in original pixel coordinates."""
    orig_w, orig_h = img.size
    with torch.no_grad():
        h_pred, v_pred = model(transform(img).unsqueeze(0).to(device))
        h_pred = h_pred.repeat_interleave(2, dim=1).squeeze(0).cpu().numpy()
        v_pred = v_pred.repeat_interleave(2, dim=1).squeeze(0).cpu().numpy()

    h_splits = np.where(
        get_middle_of_groups((h_pred > 0.5).astype(float)) == 1)[0]
    v_splits = np.where(
        get_middle_of_groups((v_pred > 0.5).astype(float)) == 1)[0]

    row_seps = [int(y * orig_h / 960.0) for y in h_splits
                if 5 < int(y * orig_h / 960.0) < orig_h - 5]
    col_seps = [int(x * orig_w / 960.0) for x in v_splits
                if 5 < int(x * orig_w / 960.0) < orig_w - 5]
    return row_seps, col_seps


def build_raw_grid(orig_w, orig_h, row_seps, col_seps):
    t = gc.Table(0, 0, orig_w, orig_h)
    for r in row_seps: t.gtRows.append(gc.Row(0, r, orig_w))
    for c in col_seps: t.gtCols.append(gc.Column(c, 0, orig_h))
    t.evaluateCells()
    return [[c.x1, c.y1, c.x2, c.y2]
            for row in t.gtCells for c in row if not c.dontCare]


# ==============================================================================
# 3. HEURISTIC MERGE PIPELINE (merge.py)
# ==============================================================================

def run_heuristic_merge(orig_w, orig_h, row_seps, col_seps, raw_ocr_list):
    """raw_ocr_list: [[id, text, x0, y0, x1, y1], ...]"""
    ocr_clean = mut.clean_ocr_data(raw_ocr_list)
    ocr_sent  = mut.make_sentences(ocr_clean)

    t = gc.Table(0, 0, orig_w, orig_h)
    for r in row_seps: t.gtRows.append(gc.Row(0, r, orig_w))
    for c in col_seps: t.gtCols.append(gc.Column(c, 0, orig_h))
    t.evaluateCells()

    merge_mod.col_merging(t, ocr_sent)
    merge_mod.row_merging(t, ocr_sent)
    merge_mod.remove_redundant_seperators(t, ocr_sent, 0.2)
    merge_mod.remove_redundant_seperators(t, ocr_sent, 0.4)
    t.merge_header_v2(ocr_sent)
    t.evaluateCells()

    return [[c.x1, c.y1, c.x2, c.y2]
            for row in t.gtCells for c in row if not c.dontCare]


# ==============================================================================
# 4. H2 ILP OPTIMIZER (milp)
# ==============================================================================

def box_intersection_area(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _tok_bbox(t):
    if isinstance(t, dict):
        return t["bbox"]
    return [t[2], t[3], t[4], t[5]]


def solve_h2_optimization(orig_w, orig_h, row_seps, col_seps, tokens):
    token_bboxes = [_tok_bbox(t) for t in tokens]
    token_areas  = [max(0.0, (b[2]-b[0])*(b[3]-b[1])) for b in token_bboxes]

    y_bounds = [0] + row_seps + [orig_h]
    x_bounds = [0] + col_seps + [orig_w]
    num_rows, num_cols = len(row_seps) + 1, len(col_seps) + 1
    num_tiles = num_rows * num_cols
    tile_to_idx = {(r, c): r * num_cols + c
                   for r in range(num_rows) for c in range(num_cols)}

    # Candidate pool
    candidates = []
    for r in range(num_rows):
        for c in range(num_cols):
            candidates.append({
                "type": "elem",
                "bbox": [x_bounds[c], y_bounds[r], x_bounds[c+1], y_bounds[r+1]],
                "tiles": [(r, c)]
            })
    for span in [2, 3]:
        for r in range(num_rows - span + 1):
            for c in range(num_cols):
                candidates.append({
                    "type": f"v_merge_{span}",
                    "bbox": [x_bounds[c], y_bounds[r],
                             x_bounds[c+1], y_bounds[r+span]],
                    "tiles": [(r+k, c) for k in range(span)]
                })
    for c in range(num_cols - 1):
        for r in range(num_rows):
            candidates.append({
                "type": "h_merge_2",
                "bbox": [x_bounds[c], y_bounds[r],
                         x_bounds[c+2], y_bounds[r+1]],
                "tiles": [(r, c), (r, c+1)]
            })

    # OCR coverage per candidate
    for cand in candidates:
        cb = cand["bbox"]
        contained = []
        for t_idx, (tb, t_area) in enumerate(zip(token_bboxes, token_areas)):
            if t_area <= 0:
                continue
            inter = box_intersection_area(tb, cb)
            if inter > 0:
                contained.append((t_idx, inter / t_area))
        cand["contained"] = contained

    # Max fraction any elementary tile achieves per token
    elem_max_frac = {}
    for t_idx, (tb, t_area) in enumerate(zip(token_bboxes, token_areas)):
        if t_area <= 0:
            elem_max_frac[t_idx] = 0.0
            continue
        mf = 0.0
        for cand in candidates:
            if cand["type"] == "elem":
                mf = max(mf, box_intersection_area(tb, cand["bbox"]) / t_area)
        elem_max_frac[t_idx] = mf

    # Weights -- identical formula for both GT and OCR
    EPSILON_PENALTY = 1.0
    weights = np.zeros(len(candidates))
    for i, cand in enumerate(candidates):
        w = 0.0
        for t, f in cand["contained"]:
            if f >= 0.70:
                w += 1.0
            elif f >= 0.50:
                w += 0.5
        if cand["type"] != "elem":
            rescued = [t for t, f in cand["contained"]
                       if f >= 0.70 and elem_max_frac.get(t, 0.0) < 0.70]
            w += (1.5 * len(rescued)) if rescued else (-EPSILON_PENALTY)
        else:
            w += 0.1
        weights[i] = max(0.001, w)

    # ILP formulation: max w^T x subject to partition constraints Ax = 1
    row_ind, col_ind, data_vals = [], [], []
    for c_idx, cand in enumerate(candidates):
        for tile in cand["tiles"]:
            row_ind.append(tile_to_idx[tile])
            col_ind.append(c_idx)
            data_vals.append(1.0)
    A = sp.csc_matrix((data_vals, (row_ind, col_ind)),
                      shape=(num_tiles, len(candidates)))

    t0 = time.perf_counter()
    res = milp(c=-weights,
               integrality=np.ones(len(candidates)),
               bounds=Bounds(0, 1),
               constraints=LinearConstraint(A,
                                            lb=np.zeros(num_tiles),
                                            ub=np.ones(num_tiles)))
    solve_ms = (time.perf_counter() - t0) * 1000.0
    x_sol = np.round(res.x).astype(int)
    opt_boxes = [candidates[i]["bbox"] for i in np.where(x_sol == 1)[0]]
    return opt_boxes, solve_ms


# ==============================================================================
# 5. EVALUATOR (IoU >= 0.25 vs PubTabNet GT)
# ==============================================================================

def compute_iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = ((b1[2]-b1[0])*(b1[3]-b1[1])
             + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter)
    return inter / union if union > 0 else 0.0


def evaluate_matching(pred_boxes, gt_boxes, thresh=0.25):
    matched_gt = set()
    for pb in pred_boxes:
        best, best_j = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = compute_iou(pb, gb)
            if iou > best:
                best, best_j = iou, j
        if best >= thresh:
            matched_gt.add(best_j)
    tp   = len(matched_gt)
    prec = tp / len(pred_boxes) if pred_boxes else 0.0
    rec  = tp / len(gt_boxes)  if gt_boxes  else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    return matched_gt, tp, prec, rec, f1


# ==============================================================================
# 6. OCR DIAGNOSTIC EVALUATION
# ==============================================================================

def ocr_coverage_diagnostic(gt_tokens, ocr_tokens, iou_thresh=0.25):
    gt_bboxes  = [_tok_bbox(t) for t in gt_tokens]
    ocr_bboxes = [[t[2], t[3], t[4], t[5]] for t in ocr_tokens]

    matched_gt, matched_ocr = set(), set()
    for j, gb in enumerate(gt_bboxes):
        for i, ob in enumerate(ocr_bboxes):
            if compute_iou(gb, ob) >= iou_thresh:
                matched_gt.add(j)
                matched_ocr.add(i)

    n_gt, n_ocr = len(gt_bboxes), len(ocr_bboxes)
    recall = len(matched_gt)  / n_gt  if n_gt  > 0 else 0.0
    prec   = len(matched_ocr) / n_ocr if n_ocr > 0 else 0.0
    f1     = 2*recall*prec/(recall+prec) if (recall+prec) > 0 else 0.0
    return {
        "gt_tokens"   : n_gt,
        "ocr_tokens"  : n_ocr,
        "matched_gt"  : len(matched_gt),
        "ocr_recall"  : round(recall, 4),
        "ocr_precision": round(prec, 4),
        "ocr_f1"      : round(f1, 4),
        "unmatched_gt": n_gt  - len(matched_gt),
        "ocr_fp"      : n_ocr - len(matched_ocr),
    }


# ==============================================================================
# 7. MAIN EXECUTION (100 TABLES)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run 100-table OCR experiment.")
    parser.add_argument("--n", type=int, default=100, help="Number of tables to evaluate")
    args = parser.parse_args()

    # Load SplitModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    model = SplitModel().to(device)
    ckpt  = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((960, 960)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    all_images = sorted(IMG_DIR.glob("*.png"))[:args.n]
    print(f"\n=======================================================")
    print(f"Starting 100-Table Controlled Experiment (N = {len(all_images)})")
    print(f"=======================================================\n")

    rows = []
    agg = {
        "gt": 0,
        "raw_cells": 0, "raw_tp": 0,
        "h1gt_cells": 0, "h1gt_tp": 0,
        "h1ocr_cells": 0, "h1ocr_tp": 0,
        "h2gt_cells": 0, "h2gt_tp": 0,
        "h2ocr_cells": 0, "h2ocr_tp": 0,
    }

    paired_h2ocr_vs_raw   = []
    paired_h2ocr_vs_h1ocr = []
    paired_h2gt_vs_h2ocr  = []
    paired_h1gt_vs_raw    = []
    paired_h1ocr_vs_raw   = []

    start_total = time.time()

    for idx, img_path in enumerate(all_images):
        stem = img_path.stem
        json_path = JSON_DIR / f"{stem}_iocr.json"
        if not json_path.exists():
            print(f"  [{idx+1:03d}] SKIP {stem}: JSON missing")
            continue

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        gt_tokens = data["pages"][0].get("tokens", [])
        gt_boxes  = [t["bbox"] for t in gt_tokens]
        raw_ocr_gt = [[t["id"], t["text"],
                       t["bbox"][0], t["bbox"][1],
                       t["bbox"][2], t["bbox"][3]]
                      for t in gt_tokens]

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Shared C_raw from SplitModel
        row_seps, col_seps = run_split_model(model, transform, img, device)
        raw_cells = build_raw_grid(orig_w, orig_h, row_seps, col_seps)

        # Realistic OCR tokens
        t_ocr = time.perf_counter()
        raw_ocr_ocr = detect_ocr_tokens(img_path)
        ocr_ms = (time.perf_counter() - t_ocr) * 1000.0

        diag = ocr_coverage_diagnostic(gt_tokens, raw_ocr_ocr)

        # Condition 1: Raw
        _, raw_tp, raw_p, raw_r, raw_f1 = evaluate_matching(raw_cells, gt_boxes)
        # Condition 2: H1-GT
        h1gt = run_heuristic_merge(orig_w, orig_h, row_seps, col_seps, raw_ocr_gt)
        _, h1gt_tp, h1gt_p, h1gt_r, h1gt_f1 = evaluate_matching(h1gt, gt_boxes)
        # Condition 3: H1-OCR
        h1ocr = run_heuristic_merge(orig_w, orig_h, row_seps, col_seps, raw_ocr_ocr)
        _, h1ocr_tp, h1ocr_p, h1ocr_r, h1ocr_f1 = evaluate_matching(h1ocr, gt_boxes)
        # Condition 4: H2-GT
        h2gt_boxes, h2gt_ms = solve_h2_optimization(orig_w, orig_h, row_seps, col_seps, gt_tokens)
        _, h2gt_tp, h2gt_p, h2gt_r, h2gt_f1 = evaluate_matching(h2gt_boxes, gt_boxes)
        # Condition 5: H2-OCR
        h2ocr_boxes, h2ocr_ms = solve_h2_optimization(orig_w, orig_h, row_seps, col_seps, raw_ocr_ocr)
        _, h2ocr_tp, h2ocr_p, h2ocr_r, h2ocr_f1 = evaluate_matching(h2ocr_boxes, gt_boxes)

        agg["gt"]          += len(gt_boxes)
        agg["raw_cells"]   += len(raw_cells)
        agg["raw_tp"]      += raw_tp
        agg["h1gt_cells"]  += len(h1gt)
        agg["h1gt_tp"]     += h1gt_tp
        agg["h1ocr_cells"] += len(h1ocr)
        agg["h1ocr_tp"]    += h1ocr_tp
        agg["h2gt_cells"]  += len(h2gt_boxes)
        agg["h2gt_tp"]     += h2gt_tp
        agg["h2ocr_cells"] += len(h2ocr_boxes)
        agg["h2ocr_tp"]    += h2ocr_tp

        paired_h2ocr_vs_raw.append(h2ocr_tp   - raw_tp)
        paired_h2ocr_vs_h1ocr.append(h2ocr_tp - h1ocr_tp)
        paired_h2gt_vs_h2ocr.append(h2gt_tp   - h2ocr_tp)
        paired_h1gt_vs_raw.append(h1gt_tp     - raw_tp)
        paired_h1ocr_vs_raw.append(h1ocr_tp   - raw_tp)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(all_images):
            print(f"[{idx+1:03d}/{len(all_images):03d}] {stem} | "
                  f"GT={len(gt_boxes):3d} | Raw={raw_tp:3d} | "
                  f"H1-GT={h1gt_tp:3d} | H1-OCR={h1ocr_tp:3d} | "
                  f"H2-GT={h2gt_tp:3d} | H2-OCR={h2ocr_tp:3d} | "
                  f"OCR Rec={diag['ocr_recall']:.2f}")

        rows.append({
            "table_id"        : stem,
            "n_gt"            : len(gt_boxes),
            "n_raw_cells"     : len(raw_cells),
            "n_grid_rows"     : len(row_seps) + 1,
            "n_grid_cols"     : len(col_seps) + 1,
            "raw_tp"          : raw_tp,
            "raw_rec"         : round(raw_r,    4),
            "raw_prec"        : round(raw_p,    4),
            "raw_f1"          : round(raw_f1,   4),
            "h1gt_tp"         : h1gt_tp,
            "h1gt_rec"        : round(h1gt_r,   4),
            "h1gt_prec"       : round(h1gt_p,   4),
            "h1gt_f1"         : round(h1gt_f1,  4),
            "h1ocr_tp"        : h1ocr_tp,
            "h1ocr_rec"       : round(h1ocr_r,  4),
            "h1ocr_prec"      : round(h1ocr_p,  4),
            "h1ocr_f1"        : round(h1ocr_f1, 4),
            "h2gt_tp"         : h2gt_tp,
            "h2gt_rec"        : round(h2gt_r,   4),
            "h2gt_prec"       : round(h2gt_p,   4),
            "h2gt_f1"         : round(h2gt_f1,  4),
            "h2ocr_tp"        : h2ocr_tp,
            "h2ocr_rec"       : round(h2ocr_r,  4),
            "h2ocr_prec"      : round(h2ocr_p,  4),
            "h2ocr_f1"        : round(h2ocr_f1, 4),
            "d_h2ocr_vs_raw"  : h2ocr_tp - raw_tp,
            "d_h2ocr_vs_h1ocr": h2ocr_tp - h1ocr_tp,
            "d_h2gt_vs_h2ocr" : h2gt_tp  - h2ocr_tp,
            "ocr_gt_tokens"   : diag["gt_tokens"],
            "ocr_n_tokens"    : diag["ocr_tokens"],
            "ocr_recall"      : diag["ocr_recall"],
            "ocr_precision"   : diag["ocr_precision"],
            "ocr_f1"          : diag["ocr_f1"],
            "ocr_unmatched_gt": diag["unmatched_gt"],
            "ocr_fp"          : diag["ocr_fp"],
            "ocr_time_ms"     : round(ocr_ms,    1),
            "h2gt_solve_ms"   : round(h2gt_ms,   1),
            "h2ocr_solve_ms"  : round(h2ocr_ms,  1),
        })

    elapsed = time.time() - start_total
    N = len(rows)
    gt_total = agg["gt"]

    # --------------------------------------------------------------------------
    # Aggregate Metrics Calculation
    # --------------------------------------------------------------------------
    def calc_metrics(tp_key, pred_key):
        tp = agg[tp_key]
        pred = agg[pred_key]
        prec = tp / pred if pred > 0 else 0.0
        rec = tp / gt_total if gt_total > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return tp, prec, rec, f1

    raw_tp, raw_prec, raw_rec, raw_f1 = calc_metrics("raw_tp", "raw_cells")
    h1gt_tp, h1gt_prec, h1gt_rec, h1gt_f1 = calc_metrics("h1gt_tp", "h1gt_cells")
    h1ocr_tp, h1ocr_prec, h1ocr_rec, h1ocr_f1 = calc_metrics("h1ocr_tp", "h1ocr_cells")
    h2gt_tp, h2gt_prec, h2gt_rec, h2gt_f1 = calc_metrics("h2gt_tp", "h2gt_cells")
    h2ocr_tp, h2ocr_prec, h2ocr_rec, h2ocr_f1 = calc_metrics("h2ocr_tp", "h2ocr_cells")

    print("\n" + "=" * 78)
    print(f"MAIN EXPERIMENT RESULTS (N = {N} tables, Total GT Cells = {gt_total})")
    print(f"Runtime: {elapsed:.1f}s ({elapsed/N*1000:.1f} ms/table)")
    print("=" * 78)
    print(f"{'Condition':<12} {'Token Source':<14} {'TP@0.25':>8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Delta vs Raw':>14}")
    print("-" * 78)
    print(f"{'Raw':<12} {'None':<14} {raw_tp:>8d} {raw_prec*100:>9.2f}% {raw_rec*100:>9.2f}% {raw_f1*100:>9.2f}% {'--':>14}")
    print(f"{'H1-GT':<12} {'GT tokens':<14} {h1gt_tp:>8d} {h1gt_prec*100:>9.2f}% {h1gt_rec*100:>9.2f}% {h1gt_f1*100:>9.2f}% {h1gt_tp - raw_tp:>+14d}")
    print(f"{'H1-OCR':<12} {'OCR tokens':<14} {h1ocr_tp:>8d} {h1ocr_prec*100:>9.2f}% {h1ocr_rec*100:>9.2f}% {h1ocr_f1*100:>9.2f}% {h1ocr_tp - raw_tp:>+14d}")
    print(f"{'H2-GT':<12} {'GT tokens':<14} {h2gt_tp:>8d} {h2gt_prec*100:>9.2f}% {h2gt_rec*100:>9.2f}% {h2gt_f1*100:>9.2f}% {h2gt_tp - raw_tp:>+14d}")
    print(f"{'H2-OCR':<12} {'OCR tokens':<14} {h2ocr_tp:>8d} {h2ocr_prec*100:>9.2f}% {h2ocr_rec*100:>9.2f}% {h2ocr_f1*100:>9.2f}% {h2ocr_tp - raw_tp:>+14d}")
    print("-" * 78)

    # --------------------------------------------------------------------------
    # Oracle-Gap Decomposition
    # --------------------------------------------------------------------------
    print("\nORACLE-GAP DECOMPOSITION:")
    print(f"  1. H2-GT  - Raw    = {h2gt_tp  - raw_tp:+d}  (Upper bound under privileged GT tokens)")
    print(f"  2. H2-OCR - Raw    = {h2ocr_tp - raw_tp:+d}  (Realistic improvement over un-repaired SplitModel)")
    print(f"  3. H2-GT  - H2-OCR = {h2gt_tp  - h2ocr_tp:+d}  (Oracle-dependence gap)")
    print(f"  4. H2-OCR - H1-OCR = {h2ocr_tp - h1ocr_tp:+d}  (Optimizer gain over heuristic repair with real OCR)")

    # --------------------------------------------------------------------------
    # Statistical Hypothesis Testing
    # --------------------------------------------------------------------------
    def run_stat_test(diffs, name):
        diffs = np.array(diffs)
        mean_d = float(np.mean(diffs))
        std_d  = float(np.std(diffs, ddof=1))
        se_d   = std_d / np.sqrt(len(diffs))
        med_d  = float(np.median(diffs))
        t_stat, t_pval = scipy.stats.ttest_1samp(diffs, 0.0)
        ci95_low = mean_d - 1.984 * se_d
        ci95_high = mean_d + 1.984 * se_d
        non_zero = diffs[diffs != 0]
        if len(non_zero) > 0:
            w_stat, w_pval = scipy.stats.wilcoxon(diffs)
        else:
            w_stat, w_pval = 0.0, 1.0
        print(f"\nStatistical Significance Test: {name} (N = {len(diffs)})")
        print(f"  Mean diff: {mean_d:+.3f} +/- {se_d:.3f} (95% CI: [{ci95_low:+.3f}, {ci95_high:+.3f}])")
        print(f"  Median diff: {med_d:+.3f}")
        print(f"  Paired t-test: t = {t_stat:.3f}, p = {t_pval:.4e}")
        print(f"  Wilcoxon signed-rank: W = {w_stat:.1f}, p = {w_pval:.4e}")
        return {
            "mean_diff": round(mean_d, 4),
            "se_diff": round(se_d, 4),
            "ci95": [round(ci95_low, 4), round(ci95_high, 4)],
            "median_diff": round(med_d, 4),
            "t_stat": round(float(t_stat), 4),
            "t_pval": float(t_pval),
            "wilcoxon_stat": round(float(w_stat), 4),
            "wilcoxon_pval": float(w_pval),
        }

    stat_h2ocr_vs_h1ocr = run_stat_test(paired_h2ocr_vs_h1ocr, "H2-OCR vs H1-OCR")
    stat_h2ocr_vs_raw   = run_stat_test(paired_h2ocr_vs_raw,   "H2-OCR vs Raw")
    stat_h2gt_vs_h2ocr  = run_stat_test(paired_h2gt_vs_h2ocr,  "H2-GT vs H2-OCR (Oracle Gap)")

    # --------------------------------------------------------------------------
    # Stratification by OCR Quality (Section 10 of Plan)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("STRATIFICATION BY OCR QUALITY (Recall vs GT)")
    print("=" * 78)
    high_tables = [r for r in rows if r["ocr_recall"] >= 0.90]
    med_tables  = [r for r in rows if 0.70 <= r["ocr_recall"] < 0.90]
    low_tables  = [r for r in rows if r["ocr_recall"] < 0.70]

    def summarize_stratum(name, t_list):
        n = len(t_list)
        if n == 0:
            print(f"  {name:<20} N = 0")
            return {}
        gt = sum(r["n_gt"] for r in t_list)
        raw = sum(r["raw_tp"] for r in t_list)
        h1gt = sum(r["h1gt_tp"] for r in t_list)
        h1ocr = sum(r["h1ocr_tp"] for r in t_list)
        h2gt = sum(r["h2gt_tp"] for r in t_list)
        h2ocr = sum(r["h2ocr_tp"] for r in t_list)
        mean_rec = np.mean([r["ocr_recall"] for r in t_list])
        print(f"  {name:<22} (N={n:2d}, Mean OCR Rec={mean_rec*100:.1f}%): "
              f"Raw={raw:4d} | H1-OCR={h1ocr:4d} | H2-OCR={h2ocr:4d} | "
              f"H2-GT={h2gt:4d} | Gain(H2-H1)={h2ocr-h1ocr:+4d}")
        return {
            "n_tables": n,
            "mean_ocr_recall": round(float(mean_rec), 4),
            "gt_cells": gt,
            "raw_tp": raw,
            "h1gt_tp": h1gt,
            "h1ocr_tp": h1ocr,
            "h2gt_tp": h2gt,
            "h2ocr_tp": h2ocr,
            "h2_vs_h1_gain": h2ocr - h1ocr,
            "oracle_gap": h2gt - h2ocr,
        }

    strat_high = summarize_stratum("High Quality (>=90%)", high_tables)
    strat_med  = summarize_stratum("Med Quality (70-90%)", med_tables)
    strat_low  = summarize_stratum("Low Quality (<70%)", low_tables)

    # --------------------------------------------------------------------------
    # Save CSV and JSON
    # --------------------------------------------------------------------------
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "n_tables": N,
        "total_gt_cells": gt_total,
        "elapsed_seconds": round(elapsed, 2),
        "conditions": {
            "Raw":    {"tp": raw_tp,    "precision": round(raw_prec, 4),    "recall": round(raw_rec, 4),    "f1": round(raw_f1, 4),    "delta_vs_raw": 0},
            "H1-GT":  {"tp": h1gt_tp,   "precision": round(h1gt_prec, 4),   "recall": round(h1gt_rec, 4),   "f1": round(h1gt_f1, 4),   "delta_vs_raw": h1gt_tp - raw_tp},
            "H1-OCR": {"tp": h1ocr_tp,  "precision": round(h1ocr_prec, 4),  "recall": round(h1ocr_rec, 4),  "f1": round(h1ocr_f1, 4),  "delta_vs_raw": h1ocr_tp - raw_tp},
            "H2-GT":  {"tp": h2gt_tp,   "precision": round(h2gt_prec, 4),   "recall": round(h2gt_rec, 4),   "f1": round(h2gt_f1, 4),   "delta_vs_raw": h2gt_tp - raw_tp},
            "H2-OCR": {"tp": h2ocr_tp,  "precision": round(h2ocr_prec, 4),  "recall": round(h2ocr_rec, 4),  "f1": round(h2ocr_f1, 4),  "delta_vs_raw": h2ocr_tp - raw_tp},
        },
        "oracle_decomposition": {
            "h2gt_minus_raw": h2gt_tp - raw_tp,
            "h2ocr_minus_raw": h2ocr_tp - raw_tp,
            "oracle_gap_h2gt_minus_h2ocr": h2gt_tp - h2ocr_tp,
            "optimizer_gain_h2ocr_minus_h1ocr": h2ocr_tp - h1ocr_tp,
        },
        "statistical_tests": {
            "h2ocr_vs_h1ocr": stat_h2ocr_vs_h1ocr,
            "h2ocr_vs_raw": stat_h2ocr_vs_raw,
            "h2gt_vs_h2ocr": stat_h2gt_vs_h2ocr,
        },
        "stratification": {
            "high": strat_high,
            "medium": strat_med,
            "low": strat_low,
        },
        "mean_ocr_recall": round(float(np.mean([r["ocr_recall"] for r in rows])), 4),
        "mean_ocr_precision": round(float(np.mean([r["ocr_precision"] for r in rows])), 4),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPer-table CSV exported to: {OUT_CSV}")
    print(f"Summary JSON exported to:    {OUT_JSON}")
    print("=" * 78)


if __name__ == "__main__":
    main()
