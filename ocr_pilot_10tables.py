"""
OCR Pilot: 10-Table GT vs OCR Token Experiment
===============================================
Runs all five conditions on the same 10 PubTabNet tables:

  Condition   Grid          Content signal   Purpose
  ---------   ----------    --------------   --------------------------------
  Raw         SplitModel    None             Baseline (no token signal)
  H1-GT       SplitModel    GT tokens        Current heuristic experiment
  H1-OCR      SplitModel    OCR tokens       Realistic heuristic
  H2-GT       SplitModel    GT tokens        Current optimizer
  H2-OCR      SplitModel    OCR tokens       Main realistic experiment

Key invariants
--------------
- Same table images for every condition.
- Same SplitModel checkpoint -> same C_raw for every row in the matrix.
- Same H1 merge code (merge.py) and H2 optimizer (solve_h2_optimization).
- Same evaluator (evaluate_matching at IoU >= 0.25, gt_boxes = PubTabNet GT).
- ONLY the token source changes: T_GT -> T_OCR.

OCR adapter
-----------
Auto-selects backend:
  1. PaddleOCR  - preferred; word-level boxes, good on small table text.
  2. OpenCV CC  - offline fallback using connected-component blob detection.
     Produces bounding boxes only (no text recognition), which is enough to
     exercise the geometric coverage weighting in H2.

Usage
-----
  python ocr_pilot_10tables.py               # first 10 cached tables
  python ocr_pilot_10tables.py --n 5         # first 5 cached tables
  python ocr_pilot_10tables.py --ocr paddle  # force PaddleOCR
  python ocr_pilot_10tables.py --ocr opencv  # force OpenCV fallback
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
OUT_CSV    = Path("d:/RRC/major_project_csvtu/ocr_pilot_results.csv")
OUT_JSON   = Path("d:/RRC/major_project_csvtu/ocr_pilot_summary.json")


# ==============================================================================
# 1. OCR ADAPTER
# ==============================================================================

def _try_paddleocr(img_path: Path):
    """Return [[id, text, x0, y0, x1, y1], ...] using PaddleOCR."""
    from paddleocr import PaddleOCR
    if not hasattr(_try_paddleocr, "_engine"):
        _try_paddleocr._engine = PaddleOCR(use_angle_cls=False, lang="en",
                                            show_log=False)
    result = _try_paddleocr._engine.ocr(str(img_path), cls=False)
    tokens = []
    if result is None or result[0] is None:
        return tokens
    for tok_id, line in enumerate(result[0]):
        pts, (text, _conf) = line
        x0 = int(min(p[0] for p in pts))
        y0 = int(min(p[1] for p in pts))
        x1 = int(max(p[0] for p in pts))
        y1 = int(max(p[1] for p in pts))
        if x1 > x0 and y1 > y0:
            tokens.append([tok_id, text, x0, y0, x1, y1])
    return tokens


def _opencv_cc_ocr(img_path: Path):
    """
    Offline OCR fallback: OpenCV connected-component blob detector.

    Binarises the image and returns each connected component's bounding box
    as a 'word' token.  Text field is a placeholder -- no recognition is done.
    Adequate for exercising the geometric coverage weighting in H2.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    # Upscale small images so CC detection is more reliable.
    scale = max(1, 400 // min(img.shape))
    if scale > 1:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_LINEAR)

    _, binary = cv2.threshold(img, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)

    img_area = img.shape[0] * img.shape[1]
    tokens = []
    tok_id = 0
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        if area < 4 or area > img_area * 0.05:
            continue
        if w < 2 or h < 2:
            continue
        x0 = int(x / scale)
        y0 = int(y / scale)
        x1 = int((x + w) / scale)
        y1 = int((y + h) / scale)
        tokens.append([tok_id, "word", x0, y0, x1, y1])
        tok_id += 1
    return tokens


def build_ocr_engine(backend: str):
    """Return a callable f(img_path) -> [[id,text,x0,y0,x1,y1],...]."""
    if backend == "paddle":
        try:
            import paddleocr  # noqa: F401
            print("OCR backend: PaddleOCR")
            return _try_paddleocr
        except ImportError:
            print("WARNING: PaddleOCR not installed; falling back to OpenCV CC.")
            return _opencv_cc_ocr
    elif backend == "opencv":
        print("OCR backend: OpenCV connected-components (offline fallback)")
        return _opencv_cc_ocr
    else:  # "auto"
        try:
            import paddleocr  # noqa: F401
            print("OCR backend: PaddleOCR (auto-selected)")
            return _try_paddleocr
        except ImportError:
            print("OCR backend: OpenCV CC (PaddleOCR not found, network issue?)")
            return _opencv_cc_ocr


# ==============================================================================
# 2. SPLIT-MODEL UTILITIES  (identical to main experiment)
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
# 3. HEURISTIC MERGE  (identical to main experiment)
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
# 4. H2 OPTIMIZER  (identical formula -- only token list changes)
# ==============================================================================

def box_intersection_area(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _tok_bbox(t):
    """Accept both dict {bbox:[...]} and list [id,text,x0,y0,x1,y1]."""
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

    # Weights -- identical formula to main experiment
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

    # ILP
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
# 5. EVALUATOR  (identical to main experiment)
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
# 6. OCR DIAGNOSTIC
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
# 7. MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",   type=int, default=10)
    parser.add_argument("--ocr", choices=["auto", "paddle", "opencv"],
                        default="auto")
    args = parser.parse_args()

    # Load SplitModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
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

    run_ocr = build_ocr_engine(args.ocr)
    all_images = sorted(IMG_DIR.glob("*.png"))[:args.n]
    print(f"\nRunning pilot on {len(all_images)} tables.\n")

    rows = []
    agg = {k: 0 for k in ["gt", "raw_tp", "h1gt_tp", "h1ocr_tp",
                            "h2gt_tp", "h2ocr_tp"]}
    paired_h2ocr_vs_raw   = []
    paired_h2ocr_vs_h1ocr = []
    paired_h2gt_vs_h2ocr  = []

    for idx, img_path in enumerate(all_images):
        stem = img_path.stem
        json_path = JSON_DIR / f"{stem}_iocr.json"
        if not json_path.exists():
            print(f"  [{idx+1}] SKIP {stem}: no JSON")
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

        # Shared C_raw
        row_seps, col_seps = run_split_model(model, transform, img, device)
        raw_cells = build_raw_grid(orig_w, orig_h, row_seps, col_seps)

        # OCR tokens
        t_ocr = time.perf_counter()
        raw_ocr_ocr = run_ocr(img_path)
        ocr_ms = (time.perf_counter() - t_ocr) * 1000.0

        diag = ocr_coverage_diagnostic(gt_tokens, raw_ocr_ocr)

        # Raw
        _, raw_tp,   raw_p,   raw_r,   raw_f1   = evaluate_matching(raw_cells, gt_boxes)
        # H1-GT
        h1gt  = run_heuristic_merge(orig_w, orig_h, row_seps, col_seps, raw_ocr_gt)
        _, h1gt_tp,  h1gt_p,  h1gt_r,  h1gt_f1  = evaluate_matching(h1gt,  gt_boxes)
        # H1-OCR
        h1ocr = run_heuristic_merge(orig_w, orig_h, row_seps, col_seps, raw_ocr_ocr)
        _, h1ocr_tp, h1ocr_p, h1ocr_r, h1ocr_f1 = evaluate_matching(h1ocr, gt_boxes)
        # H2-GT
        h2gt_boxes,  h2gt_ms  = solve_h2_optimization(orig_w, orig_h, row_seps, col_seps, gt_tokens)
        _, h2gt_tp,  h2gt_p,  h2gt_r,  h2gt_f1  = evaluate_matching(h2gt_boxes,  gt_boxes)
        # H2-OCR
        h2ocr_boxes, h2ocr_ms = solve_h2_optimization(orig_w, orig_h, row_seps, col_seps, raw_ocr_ocr)
        _, h2ocr_tp, h2ocr_p, h2ocr_r, h2ocr_f1 = evaluate_matching(h2ocr_boxes, gt_boxes)

        agg["gt"]       += len(gt_boxes)
        agg["raw_tp"]   += raw_tp
        agg["h1gt_tp"]  += h1gt_tp
        agg["h1ocr_tp"] += h1ocr_tp
        agg["h2gt_tp"]  += h2gt_tp
        agg["h2ocr_tp"] += h2ocr_tp

        paired_h2ocr_vs_raw.append(h2ocr_tp   - raw_tp)
        paired_h2ocr_vs_h1ocr.append(h2ocr_tp - h1ocr_tp)
        paired_h2gt_vs_h2ocr.append(h2gt_tp   - h2ocr_tp)

        print(f"[{idx+1:02d}] {stem}")
        print(f"       GT={len(gt_boxes):3d}  Raw={raw_tp:3d}  "
              f"H1-GT={h1gt_tp:3d}  H1-OCR={h1ocr_tp:3d}  "
              f"H2-GT={h2gt_tp:3d}  H2-OCR={h2ocr_tp:3d}")
        print(f"       OCR: n={diag['ocr_tokens']:3d}  "
              f"recall={diag['ocr_recall']:.2f}  prec={diag['ocr_precision']:.2f}  "
              f"ocr_t={ocr_ms:.0f}ms  "
              f"H2gt_t={h2gt_ms:.0f}ms  H2ocr_t={h2ocr_ms:.0f}ms")
        print()

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

    # --------------------------------------------------------------------------
    # Aggregate summary
    # --------------------------------------------------------------------------
    N  = len(rows)
    gt = agg["gt"]

    def recall_pct(tp):
        return f"{tp/gt*100:.1f}%" if gt > 0 else "--"

    print("=" * 65)
    print(f"AGGREGATE  (N={N} tables, GT cells={gt})")
    print("=" * 65)
    print(f"  {'Condition':<12} {'TP@0.25':>8}  {'Recall':>8}  {'Delta vs Raw':>13}")
    print(f"  {'-'*12} {'-'*8}  {'-'*8}  {'-'*13}")
    for label, key in [("Raw",    "raw_tp"),
                        ("H1-GT",  "h1gt_tp"),
                        ("H1-OCR", "h1ocr_tp"),
                        ("H2-GT",  "h2gt_tp"),
                        ("H2-OCR", "h2ocr_tp")]:
        tp    = agg[key]
        delta = tp - agg["raw_tp"]
        d_str = "--" if label == "Raw" else f"{delta:+d}"
        print(f"  {label:<12} {tp:>8d}  {recall_pct(tp):>8}  {d_str:>13}")

    print()
    h2gt_tp  = agg["h2gt_tp"]
    h2ocr_tp = agg["h2ocr_tp"]
    raw_tp   = agg["raw_tp"]
    print("Oracle-gap decomposition:")
    print(f"  H2-GT  - Raw     = {h2gt_tp  - raw_tp:+d}  (total potential with GT tokens)")
    print(f"  H2-OCR - Raw     = {h2ocr_tp - raw_tp:+d}  (practical gain with real OCR)")
    print(f"  H2-GT  - H2-OCR  = {h2gt_tp  - h2ocr_tp:+d}  (oracle-dependence gap)")

    if N >= 2:
        d = paired_h2ocr_vs_raw
        non_zero = [x for x in d if x != 0]
        if non_zero:
            stat, p = scipy.stats.wilcoxon(d)
            print(f"\nPaired Wilcoxon (H2-OCR vs Raw, N={N}): "
                  f"median diff={np.median(d):+.1f}, p={p:.3f}")
        else:
            print(f"\nAll paired diffs zero -- no statistical test applied.")

    ocr_recalls = [r["ocr_recall"] for r in rows]
    ocr_precs   = [r["ocr_precision"] for r in rows]
    print(f"\nOCR diagnostic averages:")
    print(f"  Mean recall    = {np.mean(ocr_recalls):.3f}")
    print(f"  Mean precision = {np.mean(ocr_precs):.3f}")

    # Save
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "n_tables"       : N,
        "ocr_backend"    : args.ocr,
        "gt_cells"       : gt,
        "raw_tp"         : raw_tp,
        "h1gt_tp"        : agg["h1gt_tp"],
        "h1ocr_tp"       : agg["h1ocr_tp"],
        "h2gt_tp"        : h2gt_tp,
        "h2ocr_tp"       : h2ocr_tp,
        "oracle_gap"     : h2gt_tp - h2ocr_tp,
        "h2ocr_vs_raw"   : h2ocr_tp - raw_tp,
        "h2ocr_vs_h1ocr" : h2ocr_tp - agg["h1ocr_tp"],
        "mean_ocr_recall": round(float(np.mean(ocr_recalls)), 4),
        "mean_ocr_prec"  : round(float(np.mean(ocr_precs)),   4),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPer-table CSV:  {OUT_CSV}")
    print(f"Summary JSON:   {OUT_JSON}")


if __name__ == "__main__":
    main()
