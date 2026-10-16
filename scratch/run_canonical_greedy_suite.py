"""
Canonical Greedy Evaluation Suite
=================================
Runs a single, frozen, unified evaluation across all 100 benchmark tables (5,646 GT cells):
- Token sources: GT vs OCR (geometric word-boxes)
- Objective: Unnormalized (w_i) vs Normalized (w_i / |tiles_i|)
- Solvers: Exact ILP (HiGHS), Greedy (Seed 42), Greedy (20 randomized tie-breaking seeds)
- Metrics: Micro/Macro Recall, Precision, F1, TEDS-S, GriTS-Top
- Per-table Jaccard similarity: Exact ILP vs Normalized Greedy cell selections
"""

import sys
import json
import csv
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import scipy.stats
from scipy.optimize import milp, LinearConstraint, Bounds
from lxml import html

BASE_DIR = Path("d:/RRC/major_project_csvtu")
sys.path.insert(0, str(BASE_DIR))

from run_full_metric_suite import (
    detect_ocr_tokens, build_candidates_and_weights, solve_milp,
    canonical_grid_to_html, evaluate_matching, box_intersection_area, box_iou
)
from teds_metric import TEDS
import grits

GRID_CACHE_FILE = BASE_DIR / "scratch/splerge_100_tables_grid_cache.json"
GT_HTML_CACHE   = BASE_DIR / "scratch/pubtabnet_100_tables_gt_html.json"
IMG_DIR         = BASE_DIR / "pubtab_samples_tokens/images"
OUT_CSV         = BASE_DIR / "canonical_greedy_suite_results.csv"
OUT_JSON        = BASE_DIR / "canonical_greedy_suite_summary.json"

NUM_SEEDS = 20

def solve_greedy_custom(candidates, weights, seed=42):
    rng = np.random.RandomState(seed)
    n = len(candidates)
    jitter = rng.uniform(0.0, 1e-6, size=n)
    effective_keys = weights + jitter
    order = np.argsort(-effective_keys)

    selected = []
    covered = set()
    for idx in order:
        cand_tiles = set(candidates[idx]["tiles"])
        if cand_tiles.isdisjoint(covered):
            covered |= cand_tiles
            selected.append(candidates[idx])
    return selected

def cell_jaccard(cells1, cells2):
    set1 = set((c["r1"], c["r2"], c["c1"], c["c2"]) for c in cells1)
    set2 = set((c["r1"], c["r2"], c["c1"], c["c2"]) for c in cells2)
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 1.0

def main():
    print("=" * 80)
    print("RUNNING CANONICAL GREEDY EVALUATION SUITE (100 TABLES)")
    print("=" * 80)

    with open(GRID_CACHE_FILE, "r") as f:
        grid_cache = json.load(f)
    with open(GT_HTML_CACHE, "r") as f:
        gt_html_cache = json.load(f)

    teds_eval = TEDS(structure_only=True)

    rows = []
    t0_start = time.perf_counter()

    for idx, (stem, g_entry) in enumerate(grid_cache.items()):
        img_path = IMG_DIR / f"{stem}.png"
        gt_entry = gt_html_cache[stem]["html_data"]

        orig_w, orig_h = g_entry["orig_w"], g_entry["orig_h"]
        row_seps, col_seps = g_entry["row_seps"], g_entry["col_seps"]
        gt_boxes = g_entry["gt_boxes"]
        gt_tokens = g_entry["gt_tokens"]
        ocr_tokens = detect_ocr_tokens(img_path)

        gt_struct_tokens = gt_entry["structure"]["tokens"]
        gt_struct_html = "<table>" + "".join(gt_struct_tokens) + "</table>"

        # Candidates & Weights: GT tokens
        c_gt, w_gt_unnorm, n_rows, n_cols, n_tiles, tile_idx = \
            build_candidates_and_weights(orig_w, orig_h, row_seps, col_seps, gt_tokens)
        w_gt_norm = np.array([w / max(1, len(c["tiles"])) for c, w in zip(c_gt, w_gt_unnorm)])

        # Candidates & Weights: OCR tokens
        c_ocr, w_ocr_unnorm, _, _, _, _ = \
            build_candidates_and_weights(orig_w, orig_h, row_seps, col_seps, ocr_tokens)
        w_ocr_norm = np.array([w / max(1, len(c["tiles"])) for c, w in zip(c_ocr, w_ocr_unnorm)])

        # ── 1. EXACT ILP ──────────────────────────────────────────────────────
        ilp_gt, _ = solve_milp(c_gt, w_gt_unnorm, n_tiles, tile_idx)
        ilp_ocr, _ = solve_milp(c_ocr, w_ocr_unnorm, n_tiles, tile_idx)

        ilp_gt_tp = evaluate_matching([c["bbox"] for c in ilp_gt], gt_boxes)[1]
        ilp_ocr_tp = evaluate_matching([c["bbox"] for c in ilp_ocr], gt_boxes)[1]

        ilp_ocr_html = canonical_grid_to_html(n_rows, n_cols, ilp_ocr)
        ilp_ocr_teds_s = teds_eval.evaluate(ilp_ocr_html, gt_struct_html)
        ilp_ocr_grits_top = grits.grits_from_html(gt_struct_html, ilp_ocr_html)["grits_top"]

        # ── 2. UNNORMALIZED GREEDY (Seed 42) ──────────────────────────────────
        g_unnorm_gt_42 = solve_greedy_custom(c_gt, w_gt_unnorm, seed=42)
        g_unnorm_ocr_42 = solve_greedy_custom(c_ocr, w_ocr_unnorm, seed=42)

        g_unnorm_gt_42_tp = evaluate_matching([c["bbox"] for c in g_unnorm_gt_42], gt_boxes)[1]
        g_unnorm_ocr_42_tp = evaluate_matching([c["bbox"] for c in g_unnorm_ocr_42], gt_boxes)[1]

        g_unnorm_ocr_html = canonical_grid_to_html(n_rows, n_cols, g_unnorm_ocr_42)
        g_unnorm_ocr_teds_s = teds_eval.evaluate(g_unnorm_ocr_html, gt_struct_html)
        g_unnorm_ocr_grits_top = grits.grits_from_html(gt_struct_html, g_unnorm_ocr_html)["grits_top"]

        # ── 3. NORMALIZED GREEDY (Seed 42) ────────────────────────────────────
        g_norm_gt_42 = solve_greedy_custom(c_gt, w_gt_norm, seed=42)
        g_norm_ocr_42 = solve_greedy_custom(c_ocr, w_ocr_norm, seed=42)

        g_norm_gt_42_tp = evaluate_matching([c["bbox"] for c in g_norm_gt_42], gt_boxes)[1]
        g_norm_ocr_42_tp = evaluate_matching([c["bbox"] for c in g_norm_ocr_42], gt_boxes)[1]

        g_norm_ocr_html = canonical_grid_to_html(n_rows, n_cols, g_norm_ocr_42)
        g_norm_ocr_teds_s = teds_eval.evaluate(g_norm_ocr_html, gt_struct_html)
        g_norm_ocr_grits_top = grits.grits_from_html(gt_struct_html, g_norm_ocr_html)["grits_top"]

        # ── 4. 20-SEED SWEEPS ────────────────────────────────────────────────
        sweep_unnorm_gt = []
        sweep_unnorm_ocr = []
        sweep_norm_gt = []
        sweep_norm_ocr = []

        for s in range(1, NUM_SEEDS + 1):
            s_u_gt = solve_greedy_custom(c_gt, w_gt_unnorm, seed=s)
            s_u_ocr = solve_greedy_custom(c_ocr, w_ocr_unnorm, seed=s)
            s_n_gt = solve_greedy_custom(c_gt, w_gt_norm, seed=s)
            s_n_ocr = solve_greedy_custom(c_ocr, w_ocr_norm, seed=s)

            sweep_unnorm_gt.append(evaluate_matching([c["bbox"] for c in s_u_gt], gt_boxes)[1])
            sweep_unnorm_ocr.append(evaluate_matching([c["bbox"] for c in s_u_ocr], gt_boxes)[1])
            sweep_norm_gt.append(evaluate_matching([c["bbox"] for c in s_n_gt], gt_boxes)[1])
            sweep_norm_ocr.append(evaluate_matching([c["bbox"] for c in s_n_ocr], gt_boxes)[1])

        # ── 5. SELECTION IDENTITY (JACCARD) ──────────────────────────────────
        jaccard_ocr = cell_jaccard(ilp_ocr, g_norm_ocr_42)
        jaccard_gt = cell_jaccard(ilp_gt, g_norm_gt_42)

        row_dict = {
            "stem": stem,
            "gt_cells": len(gt_boxes),
            "ilp_gt_tp": ilp_gt_tp,
            "ilp_ocr_tp": ilp_ocr_tp,
            "ilp_ocr_teds_s": ilp_ocr_teds_s,
            "ilp_ocr_grits_top": ilp_ocr_grits_top,
            "g_unnorm_gt_42_tp": g_unnorm_gt_42_tp,
            "g_unnorm_ocr_42_tp": g_unnorm_ocr_42_tp,
            "g_unnorm_ocr_teds_s": g_unnorm_ocr_teds_s,
            "g_unnorm_ocr_grits_top": g_unnorm_ocr_grits_top,
            "g_norm_gt_42_tp": g_norm_gt_42_tp,
            "g_norm_ocr_42_tp": g_norm_ocr_42_tp,
            "g_norm_ocr_teds_s": g_norm_ocr_teds_s,
            "g_norm_ocr_grits_top": g_norm_ocr_grits_top,
            "jaccard_ocr": jaccard_ocr,
            "jaccard_gt": jaccard_gt,
            "sweep_unnorm_gt_mean": float(np.mean(sweep_unnorm_gt)),
            "sweep_unnorm_ocr_mean": float(np.mean(sweep_unnorm_ocr)),
            "sweep_norm_gt_mean": float(np.mean(sweep_norm_gt)),
            "sweep_norm_ocr_mean": float(np.mean(sweep_norm_ocr)),
        }
        for s_idx in range(NUM_SEEDS):
            row_dict[f"unnorm_gt_s{s_idx+1}"] = sweep_unnorm_gt[s_idx]
            row_dict[f"unnorm_ocr_s{s_idx+1}"] = sweep_unnorm_ocr[s_idx]
            row_dict[f"norm_gt_s{s_idx+1}"] = sweep_norm_gt[s_idx]
            row_dict[f"norm_ocr_s{s_idx+1}"] = sweep_norm_ocr[s_idx]

        rows.append(row_dict)
        if (idx + 1) % 25 == 0:
            print(f"[{idx+1}/100] Processed in {time.perf_counter() - t0_start:.1f}s", flush=True)

    # Save CSV
    keys = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    # Compute Summary
    total_gt = sum(r["gt_cells"] for r in rows)
    unnorm_gt_seeds_totals = [sum(r[f"unnorm_gt_s{s+1}"] for r in rows) for s in range(NUM_SEEDS)]
    unnorm_ocr_seeds_totals = [sum(r[f"unnorm_ocr_s{s+1}"] for r in rows) for s in range(NUM_SEEDS)]
    norm_gt_seeds_totals = [sum(r[f"norm_gt_s{s+1}"] for r in rows) for s in range(NUM_SEEDS)]
    norm_ocr_seeds_totals = [sum(r[f"norm_ocr_s{s+1}"] for r in rows) for s in range(NUM_SEEDS)]

    summary = {
        "num_tables": len(rows),
        "total_gt_cells": total_gt,
        "exact_ilp": {
            "gt_tp": sum(r["ilp_gt_tp"] for r in rows),
            "gt_micro_rec": float(sum(r["ilp_gt_tp"] for r in rows) / total_gt),
            "ocr_tp": sum(r["ilp_ocr_tp"] for r in rows),
            "ocr_micro_rec": float(sum(r["ilp_ocr_tp"] for r in rows) / total_gt),
            "ocr_teds_s": float(np.mean([r["ilp_ocr_teds_s"] for r in rows])),
            "ocr_grits_top": float(np.mean([r["ilp_ocr_grits_top"] for r in rows]))
        },
        "unnorm_greedy_seed42": {
            "gt_tp": sum(r["g_unnorm_gt_42_tp"] for r in rows),
            "gt_micro_rec": float(sum(r["g_unnorm_gt_42_tp"] for r in rows) / total_gt),
            "ocr_tp": sum(r["g_unnorm_ocr_42_tp"] for r in rows),
            "ocr_micro_rec": float(sum(r["g_unnorm_ocr_42_tp"] for r in rows) / total_gt),
            "ocr_teds_s": float(np.mean([r["g_unnorm_ocr_teds_s"] for r in rows])),
            "ocr_grits_top": float(np.mean([r["g_unnorm_ocr_grits_top"] for r in rows]))
        },
        "unnorm_greedy_20seeds": {
            "gt_totals_mean": float(np.mean(unnorm_gt_seeds_totals)),
            "gt_totals_std": float(np.std(unnorm_gt_seeds_totals)),
            "gt_totals_min": int(np.min(unnorm_gt_seeds_totals)),
            "gt_totals_max": int(np.max(unnorm_gt_seeds_totals)),
            "ocr_totals_mean": float(np.mean(unnorm_ocr_seeds_totals)),
            "ocr_totals_std": float(np.std(unnorm_ocr_seeds_totals)),
            "ocr_totals_min": int(np.min(unnorm_ocr_seeds_totals)),
            "ocr_totals_max": int(np.max(unnorm_ocr_seeds_totals)),
            "ocr_per_seed_totals": unnorm_ocr_seeds_totals
        },
        "norm_greedy_seed42": {
            "gt_tp": sum(r["g_norm_gt_42_tp"] for r in rows),
            "gt_micro_rec": float(sum(r["g_norm_gt_42_tp"] for r in rows) / total_gt),
            "ocr_tp": sum(r["g_norm_ocr_42_tp"] for r in rows),
            "ocr_micro_rec": float(sum(r["g_norm_ocr_42_tp"] for r in rows) / total_gt),
            "ocr_teds_s": float(np.mean([r["g_norm_ocr_teds_s"] for r in rows])),
            "ocr_grits_top": float(np.mean([r["g_norm_ocr_grits_top"] for r in rows]))
        },
        "norm_greedy_20seeds": {
            "gt_totals_mean": float(np.mean(norm_gt_seeds_totals)),
            "gt_totals_std": float(np.std(norm_gt_seeds_totals)),
            "gt_totals_min": int(np.min(norm_gt_seeds_totals)),
            "gt_totals_max": int(np.max(norm_gt_seeds_totals)),
            "ocr_totals_mean": float(np.mean(norm_ocr_seeds_totals)),
            "ocr_totals_std": float(np.std(norm_ocr_seeds_totals)),
            "ocr_totals_min": int(np.min(norm_ocr_seeds_totals)),
            "ocr_totals_max": int(np.max(norm_ocr_seeds_totals)),
            "ocr_per_seed_totals": norm_ocr_seeds_totals
        },
        "selection_identity_jaccard": {
            "ocr_jaccard_mean": float(np.mean([r["jaccard_ocr"] for r in rows])),
            "ocr_jaccard_min": float(np.min([r["jaccard_ocr"] for r in rows])),
            "ocr_jaccard_max": float(np.max([r["jaccard_ocr"] for r in rows])),
            "ocr_tables_jaccard_1_0": int(sum(1 for r in rows if r["jaccard_ocr"] >= 0.9999)),
            "gt_jaccard_mean": float(np.mean([r["jaccard_gt"] for r in rows])),
            "gt_tables_jaccard_1_0": int(sum(1 for r in rows if r["jaccard_gt"] >= 0.9999))
        }
    }

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("CANONICAL GREEDY SUMMARY COMPLETED:")
    print(json.dumps(summary, indent=2))
    print("=" * 80)

if __name__ == "__main__":
    main()
