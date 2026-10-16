"""
Experiment 1a: Greedy Baseline Under Identical Objective (SPLERGE)
==================================================================
Addresses Reviewer Criticism #2 and User Directives 1 & 10.

Key updates:
1. Unit-test on EVERY table that greedy's conflict check uses the
   IDENTICAL edge set E as the ILP constraint matrix A_sparse.
2. Evaluates greedy over 20 randomized tie-break seeds to eliminate
   positional bias (mean ± range of TP gap reported).
3. Caches grid partition coordinates and tokens to scratch/splerge_100_tables_grid_cache.json
   for downstream rapid parameter sweeps.
4. Pure ASCII console output for Windows cp1252 compatibility.
"""

import sys
import os
import json
import csv
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import scipy.stats
from scipy.optimize import milp, LinearConstraint, Bounds
from PIL import Image

import torch
import torchvision.transforms as transforms

# Setup paths
BASE_DIR = Path("d:/RRC/major_project_csvtu")
SPLERGE_PATH = BASE_DIR / "AdvancedLiterateMachinery/DocumentUnderstanding/SPLERGE"
sys.path.insert(0, str(SPLERGE_PATH))

from libs.split_model_hf import SplitModel
import libs.GTElement as gc
import libs.merge_utility as mut
import merge

IMG_DIR    = BASE_DIR / "pubtab_samples_tokens/images"
JSON_DIR   = BASE_DIR / "pubtab_samples_tokens/iocr_jsons"
MODEL_PATH = SPLERGE_PATH / "model_out/split_model.pth"
OUT_CSV    = BASE_DIR / "greedy_baseline_results.csv"
OUT_JSON   = BASE_DIR / "greedy_baseline_summary.json"
CACHE_FILE = BASE_DIR / "scratch/splerge_100_tables_grid_cache.json"

NUM_SEEDS = 20


# =============================================================================
# 1. SHARED UTILITIES
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
    """Returns (matched_gt_set, tp, fp, fn, precision, recall, f1)."""
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
    fp   = len(pred_boxes) - tp
    fn   = len(gt_boxes) - tp
    prec = tp / len(pred_boxes) if pred_boxes else 0.0
    rec  = tp / len(gt_boxes)   if gt_boxes else 0.0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
    return matched_gt, tp, fp, fn, prec, rec, f1


# =============================================================================
# 2. SHARED CANDIDATE + WEIGHT BUILDER
# =============================================================================

def build_candidates_and_weights(orig_w, orig_h, row_seps, col_seps, gt_tokens,
                                 eps_penalty=1.0, rescue_mult=1.5, elem_bonus=0.1):
    y_bounds = [0] + row_seps + [orig_h]
    x_bounds = [0] + col_seps + [orig_w]
    num_rows = len(row_seps) + 1
    num_cols = len(col_seps) + 1
    num_tiles = num_rows * num_cols
    tile_to_idx = {(r, c): r * num_cols + c
                   for r in range(num_rows) for c in range(num_cols)}

    candidates = []

    # 1. Elementary candidates
    for r in range(num_rows):
        for c in range(num_cols):
            candidates.append({
                "type": "elem",
                "r1": r, "r2": r, "c1": c, "c2": c,
                "bbox": [x_bounds[c], y_bounds[r], x_bounds[c+1], y_bounds[r+1]],
                "tiles": [(r, c)]
            })

    # 2. Vertical candidate spans (height = 2, 3)
    for span_len in [2, 3]:
        for r in range(num_rows - span_len + 1):
            for c in range(num_cols):
                r_end = r + span_len - 1
                tiles = [(r_idx, c) for r_idx in range(r, r_end + 1)]
                candidates.append({
                    "type": f"v_merge_{span_len}",
                    "r1": r, "r2": r_end, "c1": c, "c2": c,
                    "bbox": [x_bounds[c], y_bounds[r], x_bounds[c+1], y_bounds[r_end+1]],
                    "tiles": tiles
                })

    # 3. Horizontal candidate spans (width = 2)
    for c in range(num_cols - 1):
        for r in range(num_rows):
            tiles = [(r, c), (r, c+1)]
            candidates.append({
                "type": "h_merge_2",
                "r1": r, "r2": r, "c1": c, "c2": c+1,
                "bbox": [x_bounds[c], y_bounds[r], x_bounds[c+2], y_bounds[r+1]],
                "tiles": tiles
            })

    token_bboxes = [tok["bbox"] for tok in gt_tokens]
    token_areas  = [(b[2]-b[0]) * (b[3]-b[1]) for b in token_bboxes]

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
                w += rescue_mult * len(rescued)
            else:
                w -= eps_penalty
        else:
            w += elem_bonus
        weights[i] = max(0.001, w)

    return candidates, weights, num_rows, num_cols, num_tiles, tile_to_idx


# =============================================================================
# 3. UNIT-TEST: IDENTICAL CONFLICT GRAPH (Greedy vs ILP)
# =============================================================================

def unit_test_conflict_graph(candidates, num_tiles, tile_to_idx):
    """
    Unit test: Verify that greedy's conflict check uses the IDENTICAL
    edge set E as the ILP constraint matrix A_sparse.
    """
    n = len(candidates)
    
    # 1. Edge set from Greedy tile-disjointness check
    edges_greedy = set()
    for i in range(n):
        tiles_i = set(tuple(t) for t in candidates[i]["tiles"])
        for j in range(i + 1, n):
            tiles_j = set(tuple(t) for t in candidates[j]["tiles"])
            if not tiles_i.isdisjoint(tiles_j):
                edges_greedy.add((i, j))
                
    # 2. Edge set from ILP constraint matrix A_sparse
    row_ind, col_ind, data_vals = [], [], []
    for c_idx, cand in enumerate(candidates):
        for tile in cand["tiles"]:
            row_ind.append(tile_to_idx[tile])
            col_ind.append(c_idx)
            data_vals.append(1.0)
    A_sparse = sp.csc_matrix((data_vals, (row_ind, col_ind)), shape=(num_tiles, n))
    
    ATA = (A_sparse.T @ A_sparse).tocoo()
    edges_ilp = set()
    for i, j, v in zip(ATA.row, ATA.col, ATA.data):
        if i < j and v > 0:
            edges_ilp.add((int(i), int(j)))
            
    assert edges_greedy == edges_ilp, (
        f"Conflict graph mismatch! Greedy edges: {len(edges_greedy)}, "
        f"ILP edges: {len(edges_ilp)}"
    )
    return len(edges_ilp), A_sparse


# =============================================================================
# 4. EXACT ILP SOLVER (HiGHS)
# =============================================================================

def solve_exact(candidates, weights, num_tiles, A_sparse):
    """Exact Maximum Weight Set Packing via ILP (HiGHS)."""
    t0 = time.perf_counter()
    res = milp(c=-weights,
               integrality=np.ones(len(candidates)),
               bounds=Bounds(0, 1),
               constraints=LinearConstraint(A_sparse,
                                            lb=np.zeros(num_tiles),
                                            ub=np.ones(num_tiles)))
    solve_ms = (time.perf_counter() - t0) * 1000.0

    x_sol = np.round(res.x).astype(int)
    selected = np.where(x_sol == 1)[0]
    opt_boxes = [candidates[i]["bbox"] for i in selected]
    opt_weight = sum(weights[i] for i in selected)

    return opt_boxes, solve_ms, opt_weight


# =============================================================================
# 5. GREEDY SOLVER OVER MULTIPLE RANDOMIZED SEEDS
# =============================================================================

def solve_greedy_seeds(candidates, weights, num_seeds=NUM_SEEDS):
    """
    Greedy baseline over multiple randomized tie-break seeds.
    Sorts by weight descending, breaking ties with uniform random permutation.
    """
    n = len(candidates)
    seed_runs = []
    
    for seed in range(num_seeds):
        t0 = time.perf_counter()
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        
        # Sort by weight descending, break ties by perm
        order = sorted(range(n), key=lambda i: (-weights[i], perm[i]))

        covered_tiles = set()
        selected_indices = []

        for idx in order:
            cand = candidates[idx]
            cand_tiles = set(tuple(t) for t in cand["tiles"])
            if cand_tiles.isdisjoint(covered_tiles):
                covered_tiles |= cand_tiles
                selected_indices.append(idx)

        solve_ms = (time.perf_counter() - t0) * 1000.0
        greedy_boxes = [candidates[i]["bbox"] for i in selected_indices]
        greedy_weight = sum(weights[i] for i in selected_indices)

        seed_runs.append({
            "seed": seed,
            "boxes": greedy_boxes,
            "solve_ms": solve_ms,
            "weight": greedy_weight,
            "count": len(greedy_boxes)
        })

    return seed_runs


# =============================================================================
# 6. MAIN EVALUATION LOOP
# =============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SplitModel on device: {device}...")
    model = SplitModel().to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((960, 960)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_images = sorted(list(IMG_DIR.glob("*.png")))
    print(f"\nGreedy vs Exact Baseline Experiment on {len(test_images)} tables (Seeds = {NUM_SEEDS})\n")

    per_table_rows = []
    grid_cache = {}

    agg = {
        "gt": 0, "raw_tp25": 0, "heur_tp25": 0, "exact_tp25": 0,
        "raw_cells": 0, "heur_cells": 0, "exact_cells": 0,
        # Greedy TP per seed (list of 20 sums)
        "greedy_tp_seeds": [0] * NUM_SEEDS,
        "greedy_cells_seeds": [0] * NUM_SEEDS,
    }

    # Per-table paired deltas for mean greedy
    paired_exact_minus_greedy_mean = []
    wlt_mean = {"wins": 0, "ties": 0, "losses": 0}

    unit_test_passed_count = 0
    start_all = time.perf_counter()

    for idx, img_path in enumerate(test_images):
        stem = img_path.stem
        json_path = JSON_DIR / f"{stem}_iocr.json"

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        orig_w = data["pages"][0]["width"]
        orig_h = data["pages"][0]["height"]
        gt_tokens = data["pages"][0]["tokens"]
        gt_boxes  = [tok["bbox"] for tok in gt_tokens if len(tok["bbox"]) == 4]

        # ── SplitModel forward pass ──────────────────────────────────────────
        img_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.size

        with torch.no_grad():
            h_pred, v_pred = model(transform(img_pil).unsqueeze(0).to(device))
            h_pred = h_pred.repeat_interleave(2, dim=1).squeeze(0).cpu().numpy()
            v_pred = v_pred.repeat_interleave(2, dim=1).squeeze(0).cpu().numpy()

        h_splits = np.where(get_middle_of_groups((h_pred > 0.5).astype(float)) == 1)[0]
        v_splits = np.where(get_middle_of_groups((v_pred > 0.5).astype(float)) == 1)[0]
        row_seps = [int(y * orig_h / 960.0) for y in h_splits if 5 < int(y * orig_h / 960.0) < orig_h - 5]
        col_seps = [int(x * orig_w / 960.0) for x in v_splits if 5 < int(x * orig_w / 960.0) < orig_w - 5]

        # Cache for downstream sensitivity sweeps and metric scripts
        grid_cache[stem] = {
            "orig_w": orig_w, "orig_h": orig_h,
            "row_seps": row_seps, "col_seps": col_seps,
            "gt_boxes": gt_boxes, "gt_tokens": gt_tokens
        }

        # ── Condition 1: Raw Split Grid ──────────────────────────────────────
        t_raw = gc.Table(0, 0, orig_w, orig_h)
        for r in row_seps: t_raw.gtRows.append(gc.Row(0, r, orig_w))
        for c in col_seps: t_raw.gtCols.append(gc.Column(c, 0, orig_h))
        t_raw.evaluateCells()
        raw_cells = [[c.x1, c.y1, c.x2, c.y2] for row in t_raw.gtCells for c in row if not c.dontCare]
        _, raw_tp25, _, _, raw_p25, raw_r25, raw_f1_25 = \
            evaluate_matching(raw_cells, gt_boxes, thresh=0.25)

        # ── Condition 2: SPLERGE Heuristic ───────────────────────────────────
        raw_ocr = [[tok["id"], tok.get("text", ""), b[0], b[1], b[2], b[3]] for tok, b in zip(gt_tokens, gt_boxes)]
        ocr_clean = mut.clean_ocr_data(raw_ocr)
        ocr_sentences = mut.make_sentences(ocr_clean)
        t_merge = gc.Table(0, 0, orig_w, orig_h)
        for r in row_seps: t_merge.gtRows.append(gc.Row(0, r, orig_w))
        for c in col_seps: t_merge.gtCols.append(gc.Column(c, 0, orig_h))
        t_merge.evaluateCells()
        merge.col_merging(t_merge, ocr_sentences)
        merge.row_merging(t_merge, ocr_sentences)
        merge.remove_redundant_seperators(t_merge, ocr_sentences, 0.2)
        merge.remove_redundant_seperators(t_merge, ocr_sentences, 0.4)
        t_merge.merge_header_v2(ocr_sentences)
        t_merge.evaluateCells()
        heur_cells = [[c.x1, c.y1, c.x2, c.y2] for row in t_merge.gtCells for c in row if not c.dontCare]
        _, heur_tp25, _, _, heur_p25, heur_r25, heur_f1_25 = \
            evaluate_matching(heur_cells, gt_boxes, thresh=0.25)

        # ── Shared candidate pool + weights ──────────────────────────────────
        candidates, weights, num_rows, num_cols, num_tiles, tile_to_idx = \
            build_candidates_and_weights(orig_w, orig_h, row_seps, col_seps, gt_tokens)

        # ── Unit Test: Exact match between Greedy and ILP conflict graphs ───
        num_conflict_edges, A_sparse = unit_test_conflict_graph(candidates, num_tiles, tile_to_idx)
        unit_test_passed_count += 1

        # ── Condition 3: H2 Exact (ILP) ─────────────────────────────────────
        exact_boxes, exact_ms, exact_weight = \
            solve_exact(candidates, weights, num_tiles, A_sparse)
        _, exact_tp25, _, _, exact_p25, exact_r25, exact_f1_25 = \
            evaluate_matching(exact_boxes, gt_boxes, thresh=0.25)

        # ── Condition 4: H2 Greedy (over 20 randomized tie-break seeds) ──────
        seed_runs = solve_greedy_seeds(candidates, weights, num_seeds=NUM_SEEDS)
        
        greedy_tps = []
        greedy_recs = []
        greedy_precs = []
        greedy_f1s = []
        greedy_weights = []
        greedy_solve_ms_list = []
        greedy_cell_counts = []

        for s_idx, s_run in enumerate(seed_runs):
            _, s_tp25, _, _, s_p25, s_r25, s_f1 = evaluate_matching(s_run["boxes"], gt_boxes, thresh=0.25)
            greedy_tps.append(s_tp25)
            greedy_recs.append(s_r25)
            greedy_precs.append(s_p25)
            greedy_f1s.append(s_f1)
            greedy_weights.append(s_run["weight"])
            greedy_solve_ms_list.append(s_run["solve_ms"])
            greedy_cell_counts.append(s_run["count"])

            agg["greedy_tp_seeds"][s_idx] += s_tp25
            agg["greedy_cells_seeds"][s_idx] += s_run["count"]

        # Statistics across seeds for this table
        g_tp_mean = float(np.mean(greedy_tps))
        g_tp_std  = float(np.std(greedy_tps))
        g_tp_min  = int(np.min(greedy_tps))
        g_tp_max  = int(np.max(greedy_tps))
        g_prec_mean = float(np.mean(greedy_precs))
        g_rec_mean  = float(np.mean(greedy_recs))
        g_f1_mean   = float(np.mean(greedy_f1s))
        g_weight_mean = float(np.mean(greedy_weights))
        g_ms_mean = float(np.mean(greedy_solve_ms_list))
        g_count_mean = float(np.mean(greedy_cell_counts))

        # ── Aggregates ───────────────────────────────────────────────────────
        agg["gt"]           += len(gt_boxes)
        agg["raw_tp25"]     += raw_tp25
        agg["heur_tp25"]    += heur_tp25
        agg["exact_tp25"]   += exact_tp25
        agg["raw_cells"]    += len(raw_cells)
        agg["heur_cells"]   += len(heur_cells)
        agg["exact_cells"]  += len(exact_boxes)

        # ── Paired deltas ────────────────────────────────────────────────────
        delta_mean = exact_tp25 - g_tp_mean
        paired_exact_minus_greedy_mean.append(delta_mean)

        if delta_mean > 0.01:
            wlt_mean["wins"] += 1
        elif delta_mean < -0.01:
            wlt_mean["losses"] += 1
        else:
            wlt_mean["ties"] += 1

        per_table_rows.append({
            "table_id": img_path.stem,
            "gt_count": len(gt_boxes),
            "grid_rows": num_rows,
            "grid_cols": num_cols,
            "num_candidates": len(candidates),
            "num_conflict_edges": num_conflict_edges,
            # Raw
            "raw_cells": len(raw_cells),
            "raw_tp25": raw_tp25,
            "raw_rec25": round(raw_r25, 4),
            "raw_prec25": round(raw_p25, 4),
            "raw_f1_25": round(raw_f1_25, 4),
            # Heuristic
            "heur_cells": len(heur_cells),
            "heur_tp25": heur_tp25,
            "heur_rec25": round(heur_r25, 4),
            "heur_prec25": round(heur_p25, 4),
            "heur_f1_25": round(heur_f1_25, 4),
            # Exact ILP
            "exact_cells": len(exact_boxes),
            "exact_tp25": exact_tp25,
            "exact_rec25": round(exact_r25, 4),
            "exact_prec25": round(exact_p25, 4),
            "exact_f1_25": round(exact_f1_25, 4),
            "exact_obj_weight": round(exact_weight, 4),
            "exact_solve_ms": round(exact_ms, 2),
            # Greedy (Mean ± Std, Min, Max over 20 seeds)
            "greedy_cells_mean": round(g_count_mean, 2),
            "greedy_tp25_mean": round(g_tp_mean, 2),
            "greedy_tp25_std": round(g_tp_std, 2),
            "greedy_tp25_min": g_tp_min,
            "greedy_tp25_max": g_tp_max,
            "greedy_rec25_mean": round(g_rec_mean, 4),
            "greedy_prec25_mean": round(g_prec_mean, 4),
            "greedy_f1_25_mean": round(g_f1_mean, 4),
            "greedy_obj_weight_mean": round(g_weight_mean, 4),
            "greedy_solve_ms_mean": round(g_ms_mean, 2),
            # Deltas
            "delta_exact_minus_greedy_tp25_mean": round(delta_mean, 2),
            "delta_exact_minus_greedy_tp25_range": [round(exact_tp25 - g_tp_max, 2), round(exact_tp25 - g_tp_min, 2)],
            "delta_exact_minus_greedy_weight": round(exact_weight - g_weight_mean, 4),
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(test_images):
            elapsed = time.perf_counter() - start_all
            g_curr_mean = np.mean(agg["greedy_tp_seeds"])
            print(f"[{idx+1:03d}/{len(test_images)}] {elapsed:.1f}s | "
                  f"Raw={agg['raw_tp25']} Heur={agg['heur_tp25']} "
                  f"Exact={agg['exact_tp25']} Greedy(mean)={g_curr_mean:.1f} | "
                  f"Exact>Greedy: W={wlt_mean['wins']} T={wlt_mean['ties']} L={wlt_mean['losses']}")

    # Save grid cache
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(grid_cache, f)
    print(f"\nGrid partition cache saved to: {CACHE_FILE}")

    # =========================================================================
    # RESULTS & STATISTICS
    # =========================================================================
    N = len(per_table_rows)
    gt_total = agg["gt"]

    # Write CSV
    fieldnames = list(per_table_rows[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_table_rows)
    print(f"Per-table CSV: {OUT_CSV}")

    # Seed aggregate stats
    g_agg_mean = float(np.mean(agg["greedy_tp_seeds"]))
    g_agg_std  = float(np.std(agg["greedy_tp_seeds"]))
    g_agg_min  = int(np.min(agg["greedy_tp_seeds"]))
    g_agg_max  = int(np.max(agg["greedy_tp_seeds"]))
    g_cells_mean = float(np.mean(agg["greedy_cells_seeds"]))

    # Statistical tests on paired delta (exact - greedy_mean)
    paired = np.array(paired_exact_minus_greedy_mean)
    nonzero = paired[np.abs(paired) > 1e-4]

    if np.std(paired) > 0:
        t_stat, p_ttest = scipy.stats.ttest_1samp(paired, 0)
    else:
        t_stat, p_ttest = 0.0, 1.0

    if len(nonzero) > 0:
        w_stat, p_wilcox = scipy.stats.wilcoxon(nonzero)
    else:
        w_stat, p_wilcox = 0.0, 1.0

    # Bootstrap 95% CI
    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(10000):
        sample = rng.choice(paired, size=N, replace=True)
        boot_means.append(sample.mean())
    boot_means = sorted(boot_means)
    ci_lo = boot_means[int(0.025 * len(boot_means))]
    ci_hi = boot_means[int(0.975 * len(boot_means))]

    def safe_pct(n, d):
        return round(n / d * 100, 2) if d > 0 else 0.0

    summary = {
        "experiment": f"Greedy vs Exact Baseline (SPLERGE, {NUM_SEEDS} randomized tie-break seeds)",
        "unit_test": {
            "identity_unit_test": "PASSED on all tables",
            "tables_tested": unit_test_passed_count,
            "description": "Conflict edge set E of greedy identically matches A_sparse^T @ A_sparse > 0 of ILP."
        },
        "num_tables": N,
        "total_gt_cells": gt_total,
        "aggregate": {
            "raw":    {"cells": agg["raw_cells"], "tp25": agg["raw_tp25"], "rec25": safe_pct(agg["raw_tp25"], gt_total)},
            "heur":   {"cells": agg["heur_cells"], "tp25": agg["heur_tp25"], "rec25": safe_pct(agg["heur_tp25"], gt_total)},
            "exact":  {"cells": agg["exact_cells"], "tp25": agg["exact_tp25"], "rec25": safe_pct(agg["exact_tp25"], gt_total)},
            "greedy": {
                "cells_mean": round(g_cells_mean, 1),
                "tp25_mean": round(g_agg_mean, 1),
                "tp25_std": round(g_agg_std, 2),
                "tp25_min": g_agg_min,
                "tp25_max": g_agg_max,
                "rec25_mean": safe_pct(g_agg_mean, gt_total),
                "per_seed_totals": agg["greedy_tp_seeds"]
            },
        },
        "exact_vs_greedy": {
            "win_loss_tie_mean": wlt_mean,
            "net_tp_gain_mean": round(agg["exact_tp25"] - g_agg_mean, 1),
            "net_tp_gain_range": [agg["exact_tp25"] - g_agg_max, agg["exact_tp25"] - g_agg_min],
            "mean_delta_per_table": round(float(np.mean(paired)), 4),
            "paired_t_stat": round(float(t_stat), 4),
            "paired_t_pvalue": round(float(p_ttest), 6),
            "wilcoxon_W": round(float(w_stat), 4),
            "wilcoxon_pvalue": round(float(p_wilcox), 6),
            "bootstrap_95CI": [round(ci_lo, 4), round(ci_hi, 4)],
        },
        "solve_time_ms": {
            "exact_mean_ms": round(float(np.mean([r["exact_solve_ms"] for r in per_table_rows])), 2),
            "exact_median_ms": round(float(np.median([r["exact_solve_ms"] for r in per_table_rows])), 2),
            "exact_p95_ms": round(float(np.percentile([r["exact_solve_ms"] for r in per_table_rows], 95)), 2),
            "greedy_mean_ms": round(float(np.mean([r["greedy_solve_ms_mean"] for r in per_table_rows])), 2),
        }
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON: {OUT_JSON}")

    # ASCII Console Output
    print("\n" + "=" * 90)
    print(f"GREEDY vs EXACT BASELINE RESULTS ({NUM_SEEDS} Randomized Tie-Break Seeds)")
    print("=" * 90)
    print(f"{'Condition':<30} | {'Cells':<7} | {'TP@0.25':<14} | {'Recall@0.25':<12}")
    print("-" * 90)
    print(f"{'1. Raw Split Grid':<30} | {agg['raw_cells']:<7} | {agg['raw_tp25']:<14} | {safe_pct(agg['raw_tp25'], gt_total):<11}%")
    print(f"{'2. SPLERGE Heuristic':<30} | {agg['heur_cells']:<7} | {agg['heur_tp25']:<14} | {safe_pct(agg['heur_tp25'], gt_total):<11}%")
    print(f"{'3. H2 Exact (ILP)':<30} | {agg['exact_cells']:<7} | {agg['exact_tp25']:<14} | {safe_pct(agg['exact_tp25'], gt_total):<11}%")
    print(f"{'4. H2 Greedy (20 seeds)':<30} | {g_cells_mean:<7.1f} | {g_agg_mean:.1f} +/- {g_agg_std:.1f}    | {safe_pct(g_agg_mean, gt_total):<11}%")
    print(f"{'   Greedy Range across seeds':<30} |         | [{g_agg_min}, {g_agg_max}]      |")
    print("=" * 90)

    print(f"\n-- THE KEY COMPARISON: Exact vs Greedy --")
    print(f"  Conflict Graph Identity: Unit test PASSED on all {unit_test_passed_count} tables (identical E)")
    print(f"  Win/Loss/Tie (mean):    {wlt_mean['wins']}W / {wlt_mean['losses']}L / {wlt_mean['ties']}T")
    print(f"  Net TP gain:            +{agg['exact_tp25'] - g_agg_mean:.1f} (range: +{agg['exact_tp25'] - g_agg_max} to +{agg['exact_tp25'] - g_agg_min})")
    print(f"  Mean delta per table:   +{np.mean(paired):.4f}")
    print(f"  Paired t-test:          t = {t_stat:.4f}, p = {p_ttest:.6e}")
    print(f"  Wilcoxon signed-rank:   W = {w_stat:.4f}, p = {p_wilcox:.6e}")
    print(f"  Bootstrap 95% CI:       [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  ILP Solve Time:         mean={summary['solve_time_ms']['exact_mean_ms']}ms, "
          f"median={summary['solve_time_ms']['exact_median_ms']}ms, "
          f"p95={summary['solve_time_ms']['exact_p95_ms']}ms")
    print("=" * 90)


if __name__ == "__main__":
    main()
