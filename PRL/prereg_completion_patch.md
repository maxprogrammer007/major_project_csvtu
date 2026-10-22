## Section 4.7 — Statistical Power, MDE, and TOST Margins (Prereg Addendum)

Target sample: N = 1000 tables (spanning-tercile strata 350/350/300, seed 20261016). Multiplicity: Holm-Bonferroni step-down, family-wise alpha = 0.05 over 5 tests (H1a, H1b, H2-TOST, H3a, H3b); the most conservative member alpha = 0.0100 is used for all a priori power statements.

### H_scale_1a — Cell-recall benefit of exact selection (H2_OCR > H1_OCR)

- Pilot (N=100): mean paired difference = +5.00 TP/table, t(99) = 6.66, implied SD of paired differences = 7.508 cells.
- Power at N=1000, alpha=0.01: 1.000000 (effect d = 0.666).
- MDE at 80% power: 0.75 TP/table; at 90% power: 0.86 TP/table (i.e., the design can detect a benefit of well under 1 cell/table).

### H_scale_1b — TEDS-S divergence (H1_OCR > H2_OCR; Branch B)

- Pilot point estimate: mean paired difference = +0.0593 TEDS-S. SD of paired differences is computed from per-table pilot data by expkit/exp10 (metric_suite_results.csv, columns h1_ocr_teds_s, h2_ocr_teds_s).

| SD of paired diffs | Power at N=1000 | MDE @80% | MDE @90% |
|---:|---:|---:|---:|
| 0.08 | 1.0000 | 0.0080 | 0.0091 |
| 0.10 | 1.0000 | 0.0100 | 0.0114 |
| 0.12 | 1.0000 | 0.0120 | 0.0137 |
| 0.15 | 1.0000 | 0.0150 | 0.0171 |
| 0.20 | 1.0000 | 0.0200 | 0.0228 |

Decision rule: if the pilot SD exceeds 0.20, the margin is retained but the paper reports the MDE rather than claiming uniform power.

### H_scale_2 — TOST equivalence of MWIS and raw TableFormer (margin ±2.0 TP)

- Margin fixed a priori at ±2.0 TP/table = 3.54% of the mean pilot table size (56.46 GT cells), justified as ambiguity within at most one 1x2 sub-cell boundary; this margin is NOT recalibrated after seeing scale data (no data-dependent margin tuning).
- Pilot: mean paired difference = -0.02 TP/table; pilot TOST p < 1e-15.

| SD of paired diffs | Power of TOST at N=1000 (alpha=0.01, one-sided members) |
|---:|---:|
| 0.5 | 1.000000 |
| 1.0 | 1.000000 |
| 2.0 | 1.000000 |
| 3.0 | 1.000000 |
| 5.0 | 1.000000 |

Conclusion: the ±2.0 TP margin retains >0.999 power for any plausible SD at N=1,000; no margin inflation is required.

### H_scale_3 — Objective design vs exactness (greedy baselines)

- H3a (normalized greedy recovers >= 95% of exact ILP cell recall): pilot = 100.0% (1,836/1,836). One-sided binomial-style test on per-table recovery ratios; with N=1,000 the 95% criterion is testable with ample precision (MDE on the mean ratio at 80% power ≈ 1.0% given any per-table SD ≤ 0.10).
- H3b (unnormalized greedy recovers < 50%): pilot = 27.45% (504/1,836; deterministic seed 42). Power > 0.999 at N=1,000 for a one-sided test of 0.50 at alpha=0.01.

### Pinned sampling fallback (pre-registered, non-adaptive)

1. Primary frame: `apoidea/pubtabnet-html`, **test split**, unique images only.
2. Draw with NumPy `default_rng(20261016)`, `choice(n_available, size=1000, replace=False)`, sorted ascending; strata assigned by spanning-tercile computed from GT HTML *before any inference*: 350 / 350 / 300 (low / mid / high fraction of spanning cells).
3. If the test split has fewer than 1,000 unique eligible images: top up from the **validation split** with the SAME PRNG stream continuing after the primary draw (no re-seeding); the split composition is logged per table.
4. A table is excluded only if (a) GT HTML fails to parse, or (b) the table region crop is unavailable. Exclusions are logged (table id + reason) and replaced by advancing the same PRNG stream; final N is reported with the exclusion count. No discretionary swaps.
5. Any deviation from this block voids the preregistration label; deviations, if forced by infrastructure failure, must be reported in the paper's reproducibility appendix.

### Interpretation guard (pre-registered)

- Confirmatory status applies ONLY to H_scale_1a..3 at N=1,000 under the frozen pipeline. All N=100 results in the main paper are labeled exploratory (protocol fixed before the multi-metric suite, but not preregistered).
- Primary endpoint for H1a/H1b is the paired per-table difference under Holm-Bonferroni; secondary (descriptive) endpoints: micro-recall, precision, F1, GriTS-Top.
