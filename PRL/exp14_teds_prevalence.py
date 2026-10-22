"""
E14 - Systematic prevalence evidence for the TEDS-S over-pruning pathology
(Branch B). NEW ANALYSIS, added after external pre-submission review.

Answers the review weakness: "two extreme examples do not establish how
often this occurs" (manuscript Sec. 4.5 / Branch B).

Pre-declared analysis grid (fixed BEFORE running; nothing is selected
post hoc - the full grid is reported):
  - destruction fraction per table:
        d_t = (h2ocr_tp - h1ocr_tp) / h2ocr_tp      (OCR token condition)
    severe-pruning thresholds D in {0.3, 0.5, 0.7}
  - high TEDS-S thresholds S in {0.8, 0.9} on the HEURISTIC score (H1_OCR)
  - paired systematic check: per-table TEDS-S advantage
        a_t = h1_ocr_teds_s - h2_ocr_teds_s
    win rate P(a_t > 0), Wilcoxon signed-rank, paired t
  - association: Pearson + Spearman between destroyed-cell count
        n_t = h2ocr_tp - h1ocr_tp
    and the TEDS-S advantage a_t (positive correlation = TEDS-S tracks
    the destruction, i.e. rewards it)

Inputs (auto-discovered in project root):
  metric_suite_results.csv  (per-table rows; columns as produced by Exp 3:
    gt_cells, raw_tp, h1gt_tp, h1ocr_tp, h2gt_tp, h2ocr_tp,
    h1_ocr_teds_s, h2_ocr_teds_s [, h1_gt_teds_s, h2_gt_teds_s ...])

Outputs -> expkit_out/exp14_teds_prevalence.json  (+ .txt report)
           expkit_out/exp14_teds_scatter.csv      (per-table scatter data,
                                                   ready for the manuscript
                                                   figure)
           expkit_out/fig14_teds_vs_destroyed.png (--plot; 300 dpi)

Usage:  python expkit/exp14_teds_prevalence.py [--plot]
Runtime: < 1 min CPU.
Exit codes: 0 ok, 2 missing input.
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

D_GRID = (0.3, 0.5, 0.7)   # severe-pruning thresholds (pre-declared)
S_GRID = (0.8, 0.9)        # high TEDS-S thresholds   (pre-declared)

report = []


def P(msg=""):
    print(msg)
    report.append(str(msg))


def main():
    csv = ROOT / "metric_suite_results.csv"
    if not csv.exists():
        hits = sorted(ROOT.glob("**/metric_suite_results.csv"))
        if not hits:
            P("[exit 2] metric_suite_results.csv not found in project root")
            (OUT / "exp14_teds_prevalence.txt").write_text("\n".join(report))
            return 2
        csv = hits[0]

    df = pd.read_csv(csv)
    c_h1tp = next((c for c in ["h1ocr_tp", "h1_ocr_tp"] if c in df.columns), None)
    c_h2tp = next((c for c in ["h2ocr_tp", "h2_ocr_tp"] if c in df.columns), None)
    c_h1s = next((c for c in ["h1_ocr_teds_s", "h1ocr_teds_s"] if c in df.columns), None)
    c_h2s = next((c for c in ["h2_ocr_teds_s", "h2ocr_teds_s"] if c in df.columns), None)

    missing = []
    if not c_h1tp: missing.append("h1ocr_tp")
    if not c_h2tp: missing.append("h2ocr_tp")
    if not c_h1s: missing.append("h1_ocr_teds_s / h1ocr_teds_s")
    if not c_h2s: missing.append("h2_ocr_teds_s / h2ocr_teds_s")
    if missing:
        P(f"[exit 2] metric_suite_results.csv missing columns: {missing}")
        (OUT / "exp14_teds_prevalence.txt").write_text("\n".join(report), encoding="utf-8")
        return 2

    n = len(df)
    h1tp = df[c_h1tp].to_numpy(float)
    h2tp = df[c_h2tp].to_numpy(float)
    h1s = df[c_h1s].to_numpy(float)
    h2s = df[c_h2s].to_numpy(float)

    destroyed = h2tp - h1tp                 # cells lost by the heuristic
    adv = h1s - h2s                         # TEDS-S advantage of heuristic
    dfrac = np.divide(destroyed, h2tp,
                      out=np.zeros_like(destroyed), where=h2tp > 0)

    P("=" * 78)
    P("E14 - TEDS-S over-pruning prevalence (Branch B, OCR condition)")
    P("=" * 78)
    P(f"input: {csv.name}  N={n} tables")
    P(f"corpus: heuristic destroys {destroyed.sum():.0f} cells in total "
      f"({destroyed.mean():+.2f}/table mean); "
      f"mean TEDS-S advantage {adv.mean():+.4f}")

    # --- 1. systematic paired comparison (not anecdotes) -------------------
    wins = int((adv > 0).sum()); ties = int((adv == 0).sum())
    losses = int((adv < 0).sum())
    w = stats.wilcoxon(h1s, h2s, zero_method="wilcox")
    t = stats.ttest_rel(h1s, h2s)
    P("")
    P("1. Paired TEDS-S, H1_OCR (heuristic) vs H2_OCR (exact):")
    P(f"   win/tie/loss = {wins}/{ties}/{losses}  "
      f"(heuristic outscores exact on {100*wins/n:.1f}% of tables)")
    P(f"   Wilcoxon p = {w.pvalue:.3e} | paired t p = {t.pvalue:.3e}")

    # --- 2. prevalence grid (all cells reported, nothing selected) ---------
    P("")
    P("2. Prevalence grid (rows = severe-pruning threshold D on the "
      "fraction of exact-selection cells destroyed by the heuristic;")
    P("   cols = heuristic TEDS-S threshold S):")
    grid = {}
    header = "   D \\ S  " + "".join(f"{'S=' + str(s):>28}" for s in S_GRID)
    P(header)
    for D in D_GRID:
        mask = dfrac >= D
        row = {"n_severe": int(mask.sum()),
               "severe_fraction": round(float(mask.mean()), 4)}
        for S in S_GRID:
            m = mask & (h1s >= S)
            row[f"n_teds_ge_{S}"] = int(m.sum())
            row[f"pct_of_corpus_teds_ge_{S}"] = round(float(m.mean()), 4)
            row[f"pct_of_severe_teds_ge_{S}"] = (
                round(float(m.sum() / mask.sum()), 4) if mask.sum() else None)
        grid[f"D={D}"] = row
        cells = "  ".join(
            f"{row[f'n_teds_ge_{S}']:>3}/{row['n_severe']:>3} severe, "
            f"{100*row[f'pct_of_corpus_teds_ge_{S}']:>5.1f}% corpus"
            for S in S_GRID)
        P(f"   {D:<7} {cells}")
    P("   (read: among tables where the heuristic destroys >=D of the exact "
      "selection's cells, n_teds_ge_S also score >=S under the heuristic)")

    # --- 3. association between destruction and TEDS-S reward --------------
    ok = h2tp > 0
    pr = stats.pearsonr(destroyed[ok], adv[ok])
    sp = stats.spearmanr(destroyed[ok], adv[ok])
    P("")
    P("3. Association (destroyed cells vs per-table TEDS-S advantage):")
    P(f"   Pearson r = {pr.statistic:+.3f} (p = {pr.pvalue:.3e})")
    P(f"   Spearman rho = {sp.statistic:+.3f} (p = {sp.pvalue:.3e})")

    # --- 4. GT-condition replication if columns exist ----------------------
    c_h1gt = next((c for c in ["h1_gt_teds_s", "h1gt_teds_s"] if c in df.columns), None)
    c_h2gt = next((c for c in ["h2_gt_teds_s", "h2gt_teds_s"] if c in df.columns), None)
    if c_h1gt and c_h2gt:
        a_gt = (df[c_h1gt] - df[c_h2gt]).to_numpy(float)
        wgt = stats.wilcoxon(df[c_h1gt], df[c_h2gt],
                             zero_method="wilcox")
        P("")
        P(f"4. GT-token condition replication: mean advantage "
          f"{a_gt.mean():+.4f}, win rate {100*(a_gt>0).mean():.1f}%, "
          f"Wilcoxon p = {wgt.pvalue:.3e}")
    else:
        P("")
        P("4. GT-token TEDS-S columns not present - OCR condition only.")

    # --- scatter export for the manuscript figure --------------------------
    sc = pd.DataFrame({
        "table": df.get("table_id", df.get("table", pd.RangeIndex(n))),
        "h1ocr_tp": h1tp.astype(int), "h2ocr_tp": h2tp.astype(int),
        "destroyed": destroyed.astype(int),
        "destruction_frac": dfrac.round(4),
        "teds_s_advantage": adv.round(4),
        "h1_ocr_teds_s": h1s.round(4), "h2_ocr_teds_s": h2s.round(4),
    })
    sc.to_csv(OUT / "exp14_teds_scatter.csv", index=False)

    # --- optional manuscript-ready figure -----------------------------------
    if "--plot" in sys.argv:
        import matplotlib.font_manager as fm
        for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
            try:
                fm.fontManager.addfont(f)
            except Exception:
                pass
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(4.6, 3.3), constrained_layout=True)
        jitter = np.zeros_like(destroyed)
        ax.scatter(destroyed + jitter, adv, s=14, alpha=0.65,
                   edgecolor="none", color="#1f4e79")
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("Cells destroyed by heuristic vs. exact selection")
        ax.set_ylabel("TEDS-S advantage of heuristic\n(H1$_\\mathrm{OCR}$ "
                      "\u2212 H2$_\\mathrm{OCR}$)")
        ax.set_title("TEDS-S advantage tracks destruction", fontsize=10)
        fig.savefig(OUT / "fig14_teds_vs_destroyed.png", dpi=300)
        P("")
        P("figure -> expkit_out/fig14_teds_vs_destroyed.png")

    result = {
        "n_tables": n,
        "paired": {"wins": wins, "ties": ties, "losses": losses,
                   "mean_advantage": float(adv.mean()),
                   "wilcoxon_p": float(w.pvalue), "paired_t_p": float(t.pvalue)},
        "prevalence_grid": grid,
        "association": {"pearson_r": float(pr.statistic),
                        "pearson_p": float(pr.pvalue),
                        "spearman_rho": float(sp.statistic),
                        "spearman_p": float(sp.pvalue)},
        "scatter_csv": str(OUT / "exp14_teds_scatter.csv"),
    }
    (OUT / "exp14_teds_prevalence.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (OUT / "exp14_teds_prevalence.txt").write_text("\n".join(report), encoding="utf-8")
    P("")
    P("outputs -> expkit_out/exp14_teds_prevalence.{json,txt}, "
      "exp14_teds_scatter.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
