"""
Experiment 2: Formalize No-Post-Processing + Naive Baselines
=============================================================
Addresses Reviewer Criticism #2.

Evaluates trivial/naive selection strategies on the SPLERGE candidate set to
establish a complete baseline ladder. All baselines use the SAME candidate pool 
and SAME tile constraints as the ILP solver.

Baselines:
  1. C_raw             : Raw split grid, no selection, no merging (already reported)
  2. Random conflict   : Randomly resolve tile conflicts (average over 100 seeds)
  3. Keep-non-conflict : Keep only elementary tiles that have no merge candidates
  4. Heuristic merge   : Published SPLERGE merge pipeline
  5. H2 Greedy         : Same weights, greedy selection
  6. H2 Exact (ILP)    : Full optimization

This runs purely on saved predictions by reloading the per-table CSV data
and re-running the lightweight selection logic. GPU is needed only for the 
SplitModel inference (shared with run_greedy_baseline.py).

Outputs:
  - naive_baselines_results.csv
  - naive_baselines_summary.json
"""

import sys
import json
import csv
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp
import scipy.stats

# ── SPLERGE imports ──────────────────────────────────────────────────────────
SPLERGE_PATH = Path("d:/RRC/major_project_csvtu/AdvancedLiterateMachinery/DocumentUnderstanding/SPLERGE")
sys.path.insert(0, str(SPLERGE_PATH))
from libs.split_model_hf import SplitModel
import libs.GTElement as gc
import libs.merge_utility as mut
import merge as merge_mod

# ── Paths ────────────────────────────────────────────────────────────────────
IMG_DIR    = Path("d:/RRC/major_project_csvtu/pubtab_samples_tokens/images")
JSON_DIR   = Path("d:/RRC/major_project_csvtu/pubtab_samples_tokens/iocr_jsons")
MODEL_PATH = SPLERGE_PATH / "model_out/split_model.pth"
OUT_CSV    = Path("d:/RRC/major_project_csvtu/naive_baselines_results.csv")
OUT_JSON   = Path("d:/RRC/major_project_csvtu/naive_baselines_summary.json")


# =============================================================================
# UTILITIES (same as run_greedy_baseline.py)
# =============================================================================

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


def box_intersection_area(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def compute_iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
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
    rec  = tp / len(gt_boxes)   if gt_boxes else 0.0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
    return tp, prec, rec, f1


# =============================================================================
# CANDIDATE + WEIGHT BUILDER (identical to run_greedy_baseline.py)
# =============================================================================

def build_candidates_and_weights(orig_w, orig_h, row_seps, col_seps, gt_tokens):
    y_bounds = [0] + row_seps + [orig_h]
    x_bounds = [0] + col_seps + [orig_w]
    num_rows = len(row_seps) + 1
    num_cols = len(col_seps) + 1
    num_tiles = num_rows * num_cols
    tile_to_idx = {(r, c): r * num_cols + c
                   for r in range(num_rows) for c in range(num_cols)}

    candidates = []
    for r in range(num_rows):
        for c in range(num_cols):
            candidates.append({
                "type": "elem",
                "bbox": [x_bounds[c], y_bounds[r], x_bounds[c+1], y_bounds[r+1]],
                "tiles": [(r, c)]
            })
    for span_len in [2, 3]:
        for r in range(num_rows - span_len + 1):
            for c in range(num_cols):
                r_end = r + span_len - 1
                candidates.append({
                    "type": f"v_merge_{span_len}",
                    "bbox": [x_bounds[c], y_bounds[r], x_bounds[c+1], y_bounds[r_end+1]],
                    "tiles": [(r_idx, c) for r_idx in range(r, r_end + 1)]
                })
    for c in range(num_cols - 1):
        for r in range(num_rows):
            candidates.append({
                "type": "h_merge_2",
                "bbox": [x_bounds[c], y_bounds[r], x_bounds[c+2], y_bounds[r+1]],
                "tiles": [(r, c), (r, c+1)]
            })

    token_bboxes = [tok["bbox"] for tok in gt_tokens]
    token_areas  = [(b[2]-b[0])*(b[3]-b[1]) for b in token_bboxes]

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
            if len(rescued) > 0:
                w += 1.5 * len(rescued)
            else:
                w -= EPSILON_PENALTY
        else:
            w += 0.1
        weights[i] = max(0.001, w)

    return candidates, weights, num_rows, num_cols, num_tiles, tile_to_idx


# =============================================================================
# NAIVE BASELINE SOLVERS
# =============================================================================

def solve_random(candidates, weights, num_tiles, tile_to_idx, n_seeds=100):
    """
    Random conflict resolution: shuffle candidates, accept if tiles not yet covered.
    Average over n_seeds random permutations.
    """
    all_tp_counts = []  # We'll accumulate boxes per seed, evaluate outside
    all_box_lists = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(candidates))
        covered = set()
        selected = []
        for idx in order:
            cand = candidates[idx]
            cand_tiles = set(tuple(t) for t in cand["tiles"])
            if cand_tiles.isdisjoint(covered):
                covered |= cand_tiles
                selected.append(cand["bbox"])
        all_box_lists.append(selected)

    return all_box_lists


def solve_elem_only(candidates, weights, num_tiles, tile_to_idx):
    """
    Keep all elementary tiles (single-cell) only. No merging at all.
    This is identical to C_raw for grid-based models.
    """
    return [c["bbox"] for c in candidates if c["type"] == "elem"]


def solve_greedy(candidates, weights, num_tiles, tile_to_idx):
    """Greedy: sort by weight desc, accept if tile-disjoint."""
    order = sorted(range(len(candidates)), key=lambda i: (-weights[i], i))
    covered = set()
    selected = []
    for idx in order:
        cand = candidates[idx]
        cand_tiles = set(tuple(t) for t in cand["tiles"])
        if cand_tiles.isdisjoint(covered):
            covered |= cand_tiles
            selected.append(cand["bbox"])
    return selected


def solve_exact(candidates, weights, num_tiles, tile_to_idx):
    """Exact ILP via HiGHS."""
    row_ind, col_ind, data_vals = [], [], []
    for c_idx, cand in enumerate(candidates):
        for tile in cand["tiles"]:
            row_ind.append(tile_to_idx[tile])
            col_ind.append(c_idx)
            data_vals.append(1.0)
    A = sp.csc_matrix((data_vals, (row_ind, col_ind)),
                      shape=(num_tiles, len(candidates)))
    res = milp(c=-weights, integrality=np.ones(len(candidates)),
               bounds=Bounds(0, 1),
               constraints=LinearConstraint(A, lb=np.zeros(num_tiles),
                                            ub=np.ones(num_tiles)))
    x_sol = np.round(res.x).astype(int)
    return [candidates[i]["bbox"] for i in np.where(x_sol == 1)[0]]


# =============================================================================
# MAIN
# =============================================================================

def main():
    CACHE_FILE = Path("d:/RRC/major_project_csvtu/scratch/splerge_100_tables_grid_cache.json")
    if CACHE_FILE.exists():
        print(f"Loading cached grid data from {CACHE_FILE}...")
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        table_items = list(cache_data.items())
    else:
        print("Cache not found, running full image evaluation...")
        table_items = []

    print(f"\nNaive Baselines Experiment on {len(table_items)} tables\n")

    per_table = []
    agg = {k: 0 for k in [
        "gt", "raw_tp", "heur_tp", "elem_tp", "random_tp_sum",
        "greedy_tp", "exact_tp",
        "raw_cells", "heur_cells", "elem_cells", "greedy_cells", "exact_cells"
    ]}
    N_RANDOM_SEEDS = 100

    for idx, (stem, cdata) in enumerate(table_items):
        orig_w = cdata["orig_w"]
        orig_h = cdata["orig_h"]
        row_seps = cdata["row_seps"]
        col_seps = cdata["col_seps"]
        gt_boxes = cdata["gt_boxes"]
        gt_tokens = cdata["gt_tokens"]
        raw_ocr = [[tok["id"], tok.get("text", ""),
                    tok["bbox"][0], tok["bbox"][1],
                    tok["bbox"][2], tok["bbox"][3]] for tok in gt_tokens]

        # Build candidates + weights
        candidates, weights, nr, nc, nt, t2i = \
            build_candidates_and_weights(orig_w, orig_h, row_seps, col_seps, gt_tokens)

        # 1. Raw (= elem only for grid model)
        raw_boxes = solve_elem_only(candidates, weights, nt, t2i)
        raw_tp, raw_p, raw_r, raw_f1 = evaluate_matching(raw_boxes, gt_boxes)

        # 2. Heuristic
        ocr_clean = mut.clean_ocr_data(raw_ocr)
        ocr_sent  = mut.make_sentences(ocr_clean)
        t_merge = gc.Table(0, 0, orig_w, orig_h)
        for r in row_seps: t_merge.gtRows.append(gc.Row(0, r, orig_w))
        for c in col_seps: t_merge.gtCols.append(gc.Column(c, 0, orig_h))
        t_merge.evaluateCells()
        merge_mod.col_merging(t_merge, ocr_sent)
        merge_mod.row_merging(t_merge, ocr_sent)
        merge_mod.remove_redundant_seperators(t_merge, ocr_sent, 0.2)
        merge_mod.remove_redundant_seperators(t_merge, ocr_sent, 0.4)
        t_merge.merge_header_v2(ocr_sent)
        t_merge.evaluateCells()
        heur_boxes = [[c.x1, c.y1, c.x2, c.y2]
                      for row in t_merge.gtCells for c in row if not c.dontCare]
        heur_tp, heur_p, heur_r, heur_f1 = evaluate_matching(heur_boxes, gt_boxes)

        # 3. Random (100 seeds)
        random_box_lists = solve_random(candidates, weights, nt, t2i, n_seeds=N_RANDOM_SEEDS)
        random_tps = []
        for rbox_list in random_box_lists:
            rtp, _, _, _ = evaluate_matching(rbox_list, gt_boxes)
            random_tps.append(rtp)
        random_mean_tp = float(np.mean(random_tps))
        random_std_tp = float(np.std(random_tps))

        # 4. Greedy
        greedy_boxes = solve_greedy(candidates, weights, nt, t2i)
        greedy_tp, greedy_p, greedy_r, greedy_f1 = evaluate_matching(greedy_boxes, gt_boxes)

        # 5. Exact ILP
        exact_boxes = solve_exact(candidates, weights, nt, t2i)
        exact_tp, exact_p, exact_r, exact_f1 = evaluate_matching(exact_boxes, gt_boxes)

        agg["gt"] += len(gt_boxes)
        agg["raw_tp"] += raw_tp
        agg["heur_tp"] += heur_tp
        agg["random_tp_sum"] += random_mean_tp
        agg["greedy_tp"] += greedy_tp
        agg["exact_tp"] += exact_tp
        agg["raw_cells"] += len(raw_boxes)
        agg["heur_cells"] += len(heur_boxes)
        agg["greedy_cells"] += len(greedy_boxes)
        agg["exact_cells"] += len(exact_boxes)

        per_table.append({
            "table_id": stem,
            "gt_count": len(gt_boxes),
            "grid": f"{nr}x{nc}",
            "n_candidates": len(candidates),
            "raw_cells": len(raw_boxes), "raw_tp": raw_tp,
            "raw_rec": round(raw_r, 4), "raw_prec": round(raw_p, 4),
            "raw_f1": round(raw_f1, 4),
            "heur_cells": len(heur_boxes), "heur_tp": heur_tp,
            "heur_rec": round(heur_r, 4), "heur_prec": round(heur_p, 4),
            "heur_f1": round(heur_f1, 4),
            "random_mean_tp": round(random_mean_tp, 2),
            "random_std_tp": round(random_std_tp, 2),
            "random_mean_rec": round(random_mean_tp / len(gt_boxes), 4) if gt_boxes else 0,
            "greedy_cells": len(greedy_boxes), "greedy_tp": greedy_tp,
            "greedy_rec": round(greedy_r, 4), "greedy_prec": round(greedy_p, 4),
            "greedy_f1": round(greedy_f1, 4),
            "exact_cells": len(exact_boxes), "exact_tp": exact_tp,
            "exact_rec": round(exact_r, 4), "exact_prec": round(exact_p, 4),
            "exact_f1": round(exact_f1, 4),
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(table_items):
            print(f"[{idx+1:03d}/{len(table_items)}] "
                  f"Raw={agg['raw_tp']} Heur={agg['heur_tp']} "
                  f"Rand={agg['random_tp_sum']:.0f} Greedy={agg['greedy_tp']} "
                  f"Exact={agg['exact_tp']}")

    # Save CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_table[0].keys()))
        writer.writeheader()
        writer.writerows(per_table)
    print(f"\nCSV saved: {OUT_CSV}")

    # Summary
    gt_total = agg["gt"]
    sp = lambda n, d: round(n/d*100, 2) if d > 0 else 0.0

    summary = {
        "experiment": "Naive Baselines + Greedy + Exact (SPLERGE)",
        "num_tables": len(per_table),
        "total_gt_cells": gt_total,
        "random_seeds": N_RANDOM_SEEDS,
        "baseline_ladder": {
            "heuristic": {
                "tp": agg["heur_tp"],
                "recall_pct": sp(agg["heur_tp"], gt_total),
            },
            "random_mean": {
                "tp": round(agg["random_tp_sum"], 1),
                "recall_pct": sp(agg["random_tp_sum"], gt_total),
            },
            "raw_elem_only": {
                "tp": agg["raw_tp"],
                "recall_pct": sp(agg["raw_tp"], gt_total),
            },
            "greedy_same_obj": {
                "tp": agg["greedy_tp"],
                "recall_pct": sp(agg["greedy_tp"], gt_total),
            },
            "exact_ilp": {
                "tp": agg["exact_tp"],
                "recall_pct": sp(agg["exact_tp"], gt_total),
            },
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON saved: {OUT_JSON}")

    # Console ladder
    print("\n" + "=" * 70)
    print("BASELINE LADDER (SPLERGE, TP@0.25)")
    print("=" * 70)
    bl = summary["baseline_ladder"]
    for name, vals in bl.items():
        tp_str = f"{vals['tp']}" if isinstance(vals['tp'], int) else f"{vals['tp']:.1f}"
        print(f"  {name:<25} TP={tp_str:>8}  Recall={vals['recall_pct']:>6}%")


if __name__ == "__main__":
    main()
