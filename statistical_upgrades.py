"""
Experiment 4a: Statistical Upgrades — TOST Equivalence Test
============================================================
Addresses Reviewer Criticism #4.

Replaces the current "p = 0.320 ⇒ not significant" framing with a proper
Two One-Sided Tests (TOST) equivalence test and reports minimum detectable
effect size. Also computes paired bootstrap CIs.

This script operates purely on existing CSV data — no model inference needed.

Inputs:
  - splerge_h2_per_table_metrics.csv  (H2 vs Raw comparison)
  - ocr_100tables_results.csv         (5-condition OCR experiment)

Output:
  - statistical_upgrades_report.json
"""

import json
import numpy as np
import pandas as pd
import scipy.stats
from pathlib import Path


BASE = Path("d:/RRC/major_project_csvtu")
OUT_JSON = BASE / "statistical_upgrades_report.json"


def tost_paired(x, y, margin):
    """
    Two One-Sided Tests (TOST) for paired equivalence.
    
    H0: |μ_x - μ_y| >= margin  (the means differ by at least `margin`)
    H1: |μ_x - μ_y| <  margin  (the means are equivalent within `margin`)
    
    Returns:
        t_lower, p_lower: test that mean diff > -margin
        t_upper, p_upper: test that mean diff < +margin
        p_tost:           max(p_lower, p_upper) — reject H0 if p_tost < α
        equivalent:       True if p_tost < 0.05
    """
    diff = np.array(x) - np.array(y)
    n = len(diff)
    mean_d = np.mean(diff)
    se = np.std(diff, ddof=1) / np.sqrt(n)

    if se == 0:
        # Zero variance: exactly equivalent if mean_d within margin
        p_tost = 0.0 if abs(mean_d) < margin else 1.0
        return 0.0, 0.0, 0.0, 0.0, p_tost, abs(mean_d) < margin

    # Lower bound test: H0: μ_d <= -margin vs H1: μ_d > -margin
    t_lower = (mean_d - (-margin)) / se
    p_lower = 1 - scipy.stats.t.cdf(t_lower, df=n-1)

    # Upper bound test: H0: μ_d >= +margin vs H1: μ_d < +margin
    t_upper = (mean_d - margin) / se
    p_upper = scipy.stats.t.cdf(t_upper, df=n-1)

    p_tost = max(p_lower, p_upper)
    equivalent = p_tost < 0.05

    return t_lower, p_lower, t_upper, p_upper, p_tost, equivalent


def bootstrap_ci(diff, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap confidence interval for mean difference."""
    rng = np.random.RandomState(seed)
    boot_means = []
    n = len(diff)
    for _ in range(n_boot):
        sample = rng.choice(diff, size=n, replace=True)
        boot_means.append(sample.mean())
    boot_means = sorted(boot_means)
    alpha = 1 - ci
    lo = boot_means[int(alpha / 2 * len(boot_means))]
    hi = boot_means[int((1 - alpha / 2) * len(boot_means))]
    return lo, hi


def compute_min_detectable_effect(diff, alpha=0.05, power=0.80):
    """
    Compute the minimum detectable effect (MDE) given observed SD and N.
    
    For a paired t-test with power `power` at significance `alpha`:
        MDE = (t_{α/2} + t_{1-power}) * SD / √n
    """
    n = len(diff)
    sd = np.std(diff, ddof=1)
    t_alpha = scipy.stats.t.ppf(1 - alpha/2, df=n-1)
    t_power = scipy.stats.t.ppf(power, df=n-1)
    mde = (t_alpha + t_power) * sd / np.sqrt(n)
    return mde


def analyze_comparison(name, x, y, margins):
    """Full statistical analysis for a paired comparison."""
    diff = np.array(x) - np.array(y)
    n = len(diff)
    mean_d = float(np.mean(diff))
    sd_d = float(np.std(diff, ddof=1))

    # Standard paired t-test (for reference)
    if sd_d > 0:
        t_stat, p_ttest = scipy.stats.ttest_1samp(diff, 0)
    else:
        t_stat, p_ttest = 0.0, 1.0

    # Wilcoxon signed-rank
    nonzero = diff[diff != 0]
    if len(nonzero) > 0:
        w_stat, p_wilcox = scipy.stats.wilcoxon(nonzero)
    else:
        w_stat, p_wilcox = 0.0, 1.0

    # Bootstrap CI
    ci_lo, ci_hi = bootstrap_ci(diff)

    # TOST at multiple margins
    tost_results = {}
    for margin in margins:
        t_l, p_l, t_u, p_u, p_tost, equiv = tost_paired(x, y, margin)
        tost_results[f"margin_{margin}"] = {
            "margin": margin,
            "t_lower": round(float(t_l), 4),
            "p_lower": round(float(p_l), 6),
            "t_upper": round(float(t_u), 4),
            "p_upper": round(float(p_u), 6),
            "p_tost": round(float(p_tost), 6),
            "equivalent_at_alpha_05": bool(equiv),
        }

    # Minimum detectable effect
    mde = compute_min_detectable_effect(diff) if sd_d > 0 else 0.0

    # Win/Loss/Tie
    wins = int(np.sum(diff > 0))
    ties = int(np.sum(diff == 0))
    losses = int(np.sum(diff < 0))

    return {
        "comparison": name,
        "n": n,
        "mean_diff": round(mean_d, 4),
        "sd_diff": round(sd_d, 4),
        "win_loss_tie": {"wins": wins, "losses": losses, "ties": ties},
        "standard_tests": {
            "paired_t_stat": round(float(t_stat), 4),
            "paired_t_pvalue": round(float(p_ttest), 6),
            "wilcoxon_W": round(float(w_stat), 4),
            "wilcoxon_pvalue": round(float(p_wilcox), 6),
        },
        "bootstrap_95CI": [round(ci_lo, 4), round(ci_hi, 4)],
        "minimum_detectable_effect": round(mde, 4),
        "tost_equivalence": tost_results,
    }


def main():
    report = {}

    # =========================================================================
    # 1. SPLERGE: H2 (Exact ILP) vs Raw Split Grid
    #    Source: splerge_h2_per_table_metrics.csv
    # =========================================================================
    csv1 = BASE / "splerge_h2_per_table_metrics.csv"
    if csv1.exists():
        df = pd.read_csv(csv1)
        print(f"Loaded {csv1.name}: {len(df)} tables")

        h2_tp = df["h2_tp_25"].values
        raw_tp = df["raw_tp_25"].values
        heur_tp = df["heur_tp_25"].values

        # H2 Exact vs Raw — the key "merging provides no value" claim
        # Margins: 1 TP, 2 TPs, 3 TPs, 5 TPs
        report["splerge_h2_exact_vs_raw"] = analyze_comparison(
            "SPLERGE H2 Exact vs Raw (TP@0.25)",
            h2_tp, raw_tp,
            margins=[1, 2, 3, 5]
        )

        # H2 Exact vs Heuristic — for completeness
        report["splerge_h2_exact_vs_heuristic"] = analyze_comparison(
            "SPLERGE H2 Exact vs Heuristic (TP@0.25)",
            h2_tp, heur_tp,
            margins=[1, 2, 5]
        )
    else:
        print(f"⚠ {csv1.name} not found, skipping SPLERGE analysis")

    # =========================================================================
    # 2. OCR Experiment: H2-OCR vs Raw, H2-OCR vs H1-OCR, etc.
    #    Source: ocr_100tables_results.csv
    # =========================================================================
    csv2 = BASE / "ocr_100tables_results.csv"
    if csv2.exists():
        df2 = pd.read_csv(csv2)
        print(f"Loaded {csv2.name}: {len(df2)} tables")

        # H2-OCR vs Raw
        report["ocr_h2ocr_vs_raw"] = analyze_comparison(
            "H2-OCR vs Raw (TP@0.25)",
            df2["h2ocr_tp"].values, df2["raw_tp"].values,
            margins=[1, 2, 3, 5]
        )

        # H2-GT vs Raw
        report["ocr_h2gt_vs_raw"] = analyze_comparison(
            "H2-GT vs Raw (TP@0.25)",
            df2["h2gt_tp"].values, df2["raw_tp"].values,
            margins=[1, 2, 3, 5]
        )

        # H2-OCR vs H1-OCR
        report["ocr_h2ocr_vs_h1ocr"] = analyze_comparison(
            "H2-OCR vs H1-OCR (TP@0.25)",
            df2["h2ocr_tp"].values, df2["h1ocr_tp"].values,
            margins=[1, 2, 5]
        )

        # H2-GT vs H2-OCR (oracle gap)
        report["ocr_h2gt_vs_h2ocr"] = analyze_comparison(
            "H2-GT vs H2-OCR (Oracle Gap, TP@0.25)",
            df2["h2gt_tp"].values, df2["h2ocr_tp"].values,
            margins=[1, 2, 3]
        )
    else:
        print(f"⚠ {csv2.name} not found, skipping OCR analysis")

    # =========================================================================
    # 3. TableFormer: MWIS vs Raw
    #    Source: tableformer_h2_metrics.csv
    # =========================================================================
    csv3 = BASE / "tableformer_h2_metrics.csv"
    if csv3.exists():
        df3 = pd.read_csv(csv3)
        print(f"Loaded {csv3.name}: {len(df3)} tables")

        report["tableformer_mwis_vs_raw"] = analyze_comparison(
            "TableFormer MWIS vs Raw (TP@0.5)",
            df3["mwis_tp"].values, df3["raw_tp"].values,
            margins=[1, 2, 3]
        )

        report["tableformer_post_vs_raw"] = analyze_comparison(
            "TableFormer Post-Heuristic vs Raw (TP@0.5)",
            df3["post_tp"].values, df3["raw_tp"].values,
            margins=[1, 2, 3]
        )
    else:
        print(f"⚠ {csv3.name} not found, skipping TableFormer analysis")

    # =========================================================================
    # SAVE REPORT
    # =========================================================================
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Statistical report saved to: {OUT_JSON}")

    # ── Console Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STATISTICAL UPGRADES REPORT")
    print("=" * 80)

    for key, result in report.items():
        print(f"\n-- {result['comparison']} --")
        print(f"  N = {result['n']}, Mean diff = {result['mean_diff']}, "
              f"SD = {result['sd_diff']}")
        wlt = result['win_loss_tie']
        print(f"  W/L/T: {wlt['wins']}W / {wlt['losses']}L / {wlt['ties']}T")
        print(f"  Paired t: t={result['standard_tests']['paired_t_stat']}, "
              f"p={result['standard_tests']['paired_t_pvalue']}")
        print(f"  Wilcoxon: W={result['standard_tests']['wilcoxon_W']}, "
              f"p={result['standard_tests']['wilcoxon_pvalue']}")
        ci = result['bootstrap_95CI']
        print(f"  Bootstrap 95% CI: [{ci[0]}, {ci[1]}]")
        print(f"  MDE (a=0.05, power=0.80): {result['minimum_detectable_effect']}")

        print(f"  TOST Equivalence Tests:")
        for mk, mv in result['tost_equivalence'].items():
            equiv_str = "[EQUIVALENT]" if mv['equivalent_at_alpha_05'] else "[NOT equivalent]"
            print(f"    Margin +/-{mv['margin']} TPs: p_TOST = {mv['p_tost']:.6f} -> {equiv_str}")


if __name__ == "__main__":
    main()
