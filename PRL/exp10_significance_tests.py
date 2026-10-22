"""
E10 — Significance tests + setup documentation numbers (NEW ANALYSIS).

Answers CV4DC Reviewer ZBKN Weakness 4:
  "The dataset, number of tables, and matching IoU for TATR is not stated and
   it also doesn't have a significance test." (and LORE "contributes little"
   -> quantify with paired tests, not narrative)

What it does
------------
A. TATR greedy-NMS vs exact MWIS:
   - per-table paired TP comparison -> wins/losses/ties
   - exact McNemar (binomial) on discordant tables
   - Wilcoxon signed-rank (zero-dropped) + paired t
   - 10,000-resample bootstrap CI of the mean TP difference
   - prints the setup block for the paper: dataset, N tables, candidate counts,
     conflict criterion (overlap ratio >= 0.25), detection score threshold.

B. LORE threshold policy (tau = 0.35 -> 0.20):
   - per-table paired recall/precision tests if the sweep CSV has per-table
     rows; otherwise aggregates only.

C. Cross-checks + prereg feed:
   - SPLERGE H2_OCR vs H1_OCR paired stats (expect 71W/0L/29T, t=6.66)
   - SD of paired TEDS-S difference (H1_OCR - H2_OCR)  -> prereg H_scale_1b
   - SD of paired (mwis_tp - raw_tp) TableFormer       -> prereg H_scale_2

Inputs (auto-discovered in project root):
  tatr_h2_per_table_metrics.csv / tatr_conflict_topology*  (Part A)
  lore_threshold_sweep_results.csv                          (Part B)
  metric_suite_results.csv                                  (Part C)
  tableformer_h2_metrics.csv                                (Part C)

Outputs -> expkit_out/exp10_significance_tests.json  (+ console report)

Usage: python expkit/exp10_significance_tests.py
Runtime: < 1 min CPU.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "expkit_out"
OUT.mkdir(exist_ok=True)

report = []


def P(msg=""):
    print(msg)
    report.append(str(msg))


def find_file(patterns):
    for pat in patterns:
        hits = sorted(ROOT.glob(pat))
        if hits:
            return hits[0]
    return None


def bootstrap_ci(diff, n=10000, seed=20261016, alpha=0.05):
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, dtype=float)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def mcnemar_exact(wins, losses):
    """Exact binomial McNemar, two-sided."""
    b, c = wins, losses
    if b + c == 0:
        return 1.0
    n = b + c
    k = min(b, c)
    p = 2.0 * sum(stats.binom.pmf(i, n, 0.5) for i in range(0, k + 1))
    return float(min(1.0, p))


def paired_block(name, a, b, higher_better=True):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = b - a
    wins = int((d > 1e-9).sum())
    losses = int((d < -1e-9).sum())
    ties = int((np.abs(d) <= 1e-9).sum())
    t_stat, t_p = stats.ttest_rel(b, a)
    try:
        w_stat, w_p = stats.wilcoxon(b, a, zero_method="wilcox")
    except Exception:
        w_stat, w_p = float("nan"), float("nan")
    lo, hi = bootstrap_ci(d)
    out = {
        "comparison": name, "n": len(d), "mean_diff": float(d.mean()),
        "wins_B": wins, "losses_B": losses, "ties": ties,
        "t_stat": float(t_stat), "t_p_two_sided": float(t_p),
        "wilcoxon_W": (None if np.isnan(w_stat) else float(w_stat)),
        "wilcoxon_p_two_sided": (None if np.isnan(w_p) else float(w_p)),
        "bootstrap95CI_mean_diff": [lo, hi],
    }
    P(f"  {name}: mean diff (B-A) = {d.mean():+.4f} | W/L/T = {wins}/{losses}/{ties} "
      f"| t = {t_stat:.3f}, p = {t_p:.3e}")
    P(f"      Wilcoxon p = {out['wilcoxon_p_two_sided'] if out['wilcoxon_p_two_sided'] is not None else float('nan'):.3e}"
      f" | bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]")
    mcn = mcnemar_exact(wins, losses)
    out["mcnemar_exact_p"] = mcn
    P(f"      McNemar exact (discordant {wins}/{losses}) p = {mcn:.4f}")
    return out


def main():
    results = {}

    # ---------------- Part A: TATR ----------------
    P("=" * 78)
    P("A. TATR — greedy NMS vs exact MWIS (setup + significance)")
    P("=" * 78)
    tatr_csv = find_file(["tatr_h2_per_table_metrics.csv", "tatr_*per_table*.csv",
                          "tatr_*metrics*.csv"])
    tatr_topology = find_file(["tatr_conflict_topology*.json", "*conflict_topology*.json"])
    setup = {
        "dataset": "PubTables-1M (test-sample crop images, 100 tables)",
        "model": "microsoft/table-transformer-structure-recognition-v1.1-all",
        "detection_score_threshold": 0.5,
        "conflict_criterion": "pairwise overlap ratio (intersection over smaller box) >= 0.25, "
                              "same axis rows/columns; components = connected components of the "
                              "conflict graph; NMS resolves each component by score-max node",
        "note": "matching IoU for TP counting = 0.25 (canonical audit matcher)",
    }
    if tatr_topology and tatr_topology.suffix == ".json":
        try:
            setup["conflict_topology"] = json.loads(tatr_topology.read_text())
        except Exception:
            pass
    results["tatr_setup"] = setup
    for k, v in setup.items():
        if k != "conflict_topology":
            P(f"  {k}: {v}")
    P("")

    if tatr_csv is None:
        P("  [warn] TATR per-table CSV not found — run the TATR consistency script first.")
        results["tatr"] = None
    else:
        df = pd.read_csv(tatr_csv)
        cols = {c.lower(): c for c in df.columns}
        greedy_col = next((cols[c] for c in cols if ("greedy" in c or "heur" in c) and "tp" in c), None)
        mwis_col = next((cols[c] for c in cols if ("mwis" in c or "exact" in c or "opt" in c) and "tp" in c), None)
        if greedy_col is None or mwis_col is None:
            P(f"  [warn] could not identify greedy/MWIS TP columns in {tatr_csv.name}: "
              f"{list(df.columns)}")
            results["tatr"] = None
        else:
            P(f"  file: {tatr_csv.name} | greedy col: {greedy_col} | MWIS col: {mwis_col}")
            results["tatr"] = {
                "file": tatr_csv.name,
                **paired_block("TATR TP: MWIS vs greedy-NMS", df[greedy_col], df[mwis_col]),
            }

    # ---------------- Part B: LORE ----------------
    P("")
    P("=" * 78)
    P("B. LORE-TSR — upstream threshold policy tau=0.35 vs tau=0.20")
    P("=" * 78)
    lore_csv = find_file(["scratch/lore_per_table_sweep_results.csv", "lore_threshold_sweep_results.csv", "lore*sweep*.csv"])
    if lore_csv is None:
        P("  [warn] LORE sweep CSV not found.")
        results["lore"] = None
    else:
        df = pd.read_csv(lore_csv)
        P(f"  file: {lore_csv.name} | columns: {list(df.columns)[:12]}"
          + (" ..." if len(df.columns) > 12 else ""))
        has_table_col = any("table" in c.lower() or "stem" in c.lower() or "id" in c.lower()
                            for c in df.columns)
        has_tau = any("tau" in c.lower() or "thresh" in c.lower() or "conf" in c.lower()
                      for c in df.columns)
        if has_table_col and has_tau:
            tcol = next(c for c in df.columns if "tau" in c.lower()
                        or "thresh" in c.lower() or "conf" in c.lower())
            taus = sorted(df[tcol].unique())
            results["lore"] = {"file": lore_csv.name, "taus": [float(t) for t in taus]}
            for metric, pat in [("tp", "tp"), ("recall", "rec"), ("precision", "prec"), ("f1", "f1")]:
                mcols = [c for c in df.columns if pat in c.lower() and c.lower() != "table_id"]
                if not mcols:
                    continue
                mcol = mcols[0]
                wide = df.pivot_table(index=df.columns[0], columns=tcol, values=mcol)
                for a, b in [(0.35, 0.20)]:
                    if a in wide.columns and b in wide.columns:
                        P(f"  metric: {mcol}")
                        results["lore"][f"{mcol}_{a}_vs_{b}"] = paired_block(
                            f"LORE {mcol}: tau={b} vs tau={a}", wide[a].values, wide[b].values)
        else:
            P("  [warn] sweep CSV lacks per-table rows; aggregate table goes in the paper as-is "
              "(tau=0.35: recall 31.19%/precision 99.04% -> tau=0.20: 90.52%/99.18%). "
              "Re-run sweep_lore_thresholds.py with per-table output for paired tests.")
            results["lore"] = {"file": lore_csv.name, "per_table": False}

    # ---------------- Part C: cross-checks + prereg feed ----------------
    P("")
    P("=" * 78)
    P("C. Cross-checks + prereg SD extraction")
    P("=" * 78)
    ms_csv = find_file(["metric_suite_results.csv"])
    if ms_csv is not None:
        df = pd.read_csv(ms_csv)
        total_gt = int(df["gt_cells"].sum())
        P(f"  metric_suite_results.csv: N={len(df)} tables, total GT cells={total_gt}")
        P(f"  raw={int(df['raw_tp'].sum())} h1gt={int(df['h1gt_tp'].sum())} "
          f"h1ocr={int(df['h1ocr_tp'].sum())} h2gt={int(df['h2gt_tp'].sum())} "
          f"h2ocr={int(df['h2ocr_tp'].sum())}")

        results["splerge"] = {
            "n_tables": len(df), "total_gt_cells": total_gt,
            "raw_tp": int(df["raw_tp"].sum()),
            "h1_gt_tp": int(df["h1gt_tp"].sum()),
            "h1_ocr_tp": int(df["h1ocr_tp"].sum()),
            "h2_gt_tp": int(df["h2gt_tp"].sum()),
            "h2_ocr_tp": int(df["h2ocr_tp"].sum()),
            "paired_h2ocr_vs_h1ocr": paired_block(
                "SPLERGE cell TP: H2_OCR (ILP) vs H1_OCR (heuristic)",
                df["h1ocr_tp"], df["h2ocr_tp"]),
        }

        if "h1_ocr_teds_s" in df.columns and "h2_ocr_teds_s" in df.columns:
            d_teds = (df["h1_ocr_teds_s"] - df["h2_ocr_teds_s"]).values
            P("")
            P(f"  prereg H_scale_1b: SD of paired TEDS-S diff (H1_OCR - H2_OCR) = "
              f"{d_teds.std(ddof=1):.4f} | mean = {d_teds.mean():+.4f}")
            results["prereg_h1b"] = {
                "sd_paired_teds_diff": float(d_teds.std(ddof=1)),
                "mean_paired_teds_diff": float(d_teds.mean()),
            }
            paired_block("TEDS-S: H1_OCR vs H2_OCR (Branch B)",
                         df["h2_ocr_teds_s"], df["h1_ocr_teds_s"])
    else:
        P("  [warn] metric_suite_results.csv not found")

    tf_csv = find_file(["scratch/tableformer_mwis_factorial_results.csv", "*tableformer*mwis*factorial*.csv", "tableformer_h2_metrics.csv"])
    if tf_csv is not None:
        df = pd.read_csv(tf_csv)
        mcol = "mwis_ocr_tp" if "mwis_ocr_tp" in df.columns else ("mwis_tp" if "mwis_tp" in df.columns else None)
        rcol = "raw_tp" if "raw_tp" in df.columns else None
        if mcol and rcol:
            d = (df[mcol] - df[rcol]).values
            P("")
            P(f"  {tf_csv.name}: N={len(df)} | raw={int(df[rcol].sum())} "
              f"post={int(df['post_tp'].sum()) if 'post_tp' in df.columns else '?'} "
              f"mwis={int(df[mcol].sum())}")
            P(f"  prereg H_scale_2: SD of paired (mwis - raw) TP = {d.std(ddof=1):.4f} "
              f"| mean = {d.mean():+.4f}")
            results["prereg_h2"] = {
                "n_tables": len(df),
                "raw_tp": int(df[rcol].sum()),
                "mwis_tp": int(df[mcol].sum()),
                "sd_paired_diff": float(d.std(ddof=1)),
                "mean_paired_diff": float(d.mean()),
            }
            results["tableformer_tost"] = paired_block(
                "TableFormer cell TP: MWIS vs raw", df[rcol], df[mcol])

    out_json = OUT / "exp10_significance_tests.json"
    out_json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (OUT / "exp10_significance_tests.txt").write_text("\n".join(report), encoding="utf-8")
    P("")
    P(f"[E10] done -> {out_json}")


if __name__ == "__main__":
    main()
