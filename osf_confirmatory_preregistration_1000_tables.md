# Confirmatory Study Pre-Registration: Post-Processing Artifacts and Combinatorial Optimization in Table Structure Recognition (1,000-Table Scale Study)

**Standard Template:** AsPredicted / OSF Standard Pre-Registration  
**Date of Pre-Registration:** October 17, 2026  
**Status:** PROSPECTIVE PRE-REGISTRATION (Fixed in writing prior to sampling and execution of 1,000-table inference)  
**Authors:** Research Team (CSVTU Major Project)  
**Repository State:** Git Commit `cdcda075c37aaaf1499be92667ea7aeac9a750c3`

---

## 1. Data Collection, Sampling Frame & Benchmark Scope

- **Sampling Frame & Source Dataset:** HuggingFace repository [`apoidea/pubtabnet-html`](https://huggingface.co/datasets/apoidea/pubtabnet-html) (`test` split; extended with `validation` split if necessary to guarantee $N = 1,000$ unique valid table instances).
- **Target Sample Size:** Exactly $N = 1,000$ distinct tables.
- **Random Sampling Seed:** Seed `20261016` using Python's standard pseudo-random number generator (`random.Random(20261016)`).
- **Stratification Scheme (Spanning Fraction Terciles):** To guarantee balanced structural representation across table complexities, the sample is prospectively stratified into three terciles based on ground-truth spanning-cell fraction:
  1. *Low Spanning Tercile (< 10% spanning cells):* Exactly $N = 350$ tables.
  2. *Medium Spanning Tercile (10% – 25% spanning cells):* Exactly $N = 350$ tables.
  3. *High Spanning Tercile (> 25% spanning cells):* Exactly $N = 300$ tables.
- **Cell Count Distribution Control:** The total cell count per table is recorded across complexity bins ($\le 25$ cells, $26–60$ cells, $> 60$ cells) to confirm uniform complexity coverage across splits.
- **Fixed Global Denominators:** Total ground-truth cells across all 1,000 sampled tables serve as the invariant denominator for all micro-average metrics.

---

## 2. Matcher Protocol & Ambiguity Justification

- **Canonical Matcher Protocol:** All cell matching across all paradigms and conditions is performed via **canonical global bipartite maximum-IoU greedy matching** (`np.argsort(iou_matrix.ravel())[::-1]`). An IoU threshold of $\ge 0.50$ is enforced for True Positive assignment (with $\ge 0.25$ recorded as a secondary sensitivity check).
- **Equivalence Verification to Official LORE `pairTab`:** The canonical global bipartite matcher has been empirically demonstrated to be 100.00% equivalent to the official verbatim `pairTab` matching loop (`AdvancedLiterateMachinery/.../LORE-TSR/src/lib/utils/eval_utils.py`) across all 100 tables in the canonical benchmark corpus (SHA-256: `b5ebe3c25f948db06722eccbd303bda8c5293e58da5e858b01cbcc6ecefdb903`):
  - At $\tau = 0.35$: Exactly 1,761 TPs across both matchers (100/100 tables identical, 1,761/1,761 pairs identical = 100.00%).
  - At $\tau = 0.20$: Exactly 5,111 TPs across both matchers (100/100 tables identical, 5,111/5,111 pairs identical = 100.00%).
  - At $\tau = 0.01$: Exactly 5,180 TPs across both matchers (100/100 tables identical, 5,180/5,180 pairs identical = 100.00%).
- **LORE Degeneracy Justification (Assignment Ambiguity Count):** In the 100-table canonical benchmark (5,646 ground truth cells), the count of tables with any $\ge 2$-candidate assignment ambiguity (cells with multiple candidate bounding boxes having $\text{IoU} \ge 0.50$) is **0 out of 100 tables** (0 of 5,646 GT cells) at both operational thresholds ($\tau = 0.35$ and $\tau = 0.20$). Because CenterNet's anchor-free point detection produces non-overlapping local maxima, competing candidate clusters do not exist, rendering sequential first-match, score-sorted greedy, and global bipartite matching mathematically degenerate and strictly identical.

---

## 3. Dependent Variables, Metric Definitions & Implementation Details

1. **Primary Geometric Metric (Cell True Positives & Micro-Recall):** True Positive cell detections matched via canonical bipartite matching at $\text{IoU} \ge 0.50$ and $\ge 0.25$. Micro-Recall is defined as $\sum \text{TP} / \sum \text{GT}$; Macro-Recall is the unweighted mean across tables.
2. **Primary Structural Metric (TEDS-S):** Tree Edit Distance Structure-Only computed via the official CC-OCR evaluation script (`eval_teds.py`) utilizing the APTED 1.0.3 algorithm, with HTML tree parser configuration explicitly pinning:
   ```python
   ignore_nodes = ["thead", "tbody"]
   ```
3. **HTML Conversion Protocol:** Predictions are converted to HTML tables using the canonical serialization protocol (`_tokens_to_canonical_html` and `html_table_parser`) ensuring identical tree tags across all comparative conditions.
4. **Primary Topological Metric (GriTS-Top):** Grid Table Similarity Topology computed via the official Microsoft Table Transformer evaluator (`src/grits.py` from `microsoft/table-transformer`, frozen in `d:/RRC/major_project_csvtu/grits.py`).
5. **Secondary Content Metric (GriTS-Con):** Strictly evaluated only on genuine text/token conditions; strictly omitted on geometric word-box conditions due to placeholder string artifact collapse.
6. **Efficiency & Scalability:** Solver runtime per table in milliseconds (mean, median, 95th percentile).

---

## 4. Canonical Benchmark Baselines (Frozen 100-Table Reference)

All confirmatory hypotheses are prospectively anchored to the verified canonical baselines from the frozen 100-table benchmark ($N = 100$, 5,646 GT cells):

- **SPLERGE Split-and-Merge (5,646 GT Cells):**
  - Raw Split Grid: 1,833 TPs (32.47% micro-recall)
  - H1 Heuristic Post-Processing (Geometric Word-Boxes): **1,336 TPs** (23.66% micro-recall, -27.11% destruction)
  - H1 Heuristic Post-Processing (Oracle GT Tokens): 1,203 TPs (21.31% micro-recall, -34.37% destruction)
  - H2 Exact Set Packing ILP (Geometric Word-Boxes): **1,836 TPs** (32.52% micro-recall, **+500 net TPs over H1**, 71 wins, 29 ties, 0 losses)
  - H2 Exact Set Packing ILP (Oracle GT Tokens): **1,839 TPs** (32.57% micro-recall; +3 over word-boxes: 95 ties, 3 GT wins, 2 OCR wins, Wilcoxon $W = 5.0, p = 0.4795$)
  - Standard Greedy (Unnormalized, Word-Boxes): 504 TPs (8.93% micro-recall)
  - Normalized Greedy (Footprint-Normalized, Word-Boxes): 1,835 TPs (32.50% micro-recall, mean Jaccard = 0.9962; 93/100 identical solutions)
  - Branch B Metric Divergence Baseline: H1 TEDS-S **0.7318** vs. H2 TEDS-S **0.6725** ($p = 0.0044$); H1 GriTS-Top **0.7908** vs. H2 GriTS-Top **0.7605**.
- **TableFormer Continuous Regression (5,646 GT Cells, 3×2 Matrix Baseline):**
  - Raw Continuous Regressor: **5,204 TPs** (92.17% micro-recall) [Token-independent]
  - Heuristic Post-Processing Snap OFF (Realistic OCR Word-Boxes): **4,921 TPs** (87.16% micro-recall, -283 below raw)
  - Heuristic Post-Processing Snap ON (Realistic OCR Word-Boxes): **3,902 TPs** (69.11% micro-recall, -1,302 below raw; -1,019 snapping destruction)
  - Heuristic Post-Processing Snap ON (Oracle GT Tokens - Published Setup): **5,488 TPs** (97.20% micro-recall, +406 circular inflation)
  - Heuristic Post-Processing Snap OFF (Oracle GT Tokens): **5,082 TPs** (89.99% micro-recall, -122 below raw)
  - Exact MWIS Selection (Realistic OCR Word-Boxes): **5,202 TPs** (92.14% micro-recall; **rescues +281 cells over de-snapped heuristic**, $t = 4.28, p = 4.36 \times 10^{-5}$; 42 wins, 55 ties, 3 losses)
  - Exact MWIS Selection (Oracle GT Tokens): **5,203 TPs** (92.15% micro-recall; exactly 1-cell gap from OCR MWIS)
  - Equivalence: MWIS (OCR) and Raw are statistically equivalent within $\pm 2.0$ TPs ($p_{\text{TOST}} = 5.46 \times 10^{-117}$).

---

## 5. Registered Confirmatory Hypotheses & Directional Commitments

We commit prospectively to reporting the outcomes of all four confirmatory hypotheses whichever way the empirical data lands:

### Hypothesis $H_{\text{scale}, 1a}$ (SPLERGE Cell Recall Recovery)
- **Prediction:** Exact Maximum Weight Set Packing (H2) on realistic geometric word-boxes will achieve a statistically significant increase in cell True Positives over SPLERGE's heuristic post-processing (H1) ($p < 0.001$, paired Wilcoxon signed-rank test and paired $t$-test), recovering $\ge 90\%$ of cells destroyed by the heuristic relative to the Raw Split Grid baseline (scaling the baseline win of $+500$ TPs on the 100-table benchmark: $1,336 \rightarrow 1,836$ TPs).
- **Decision Rule:** Reject null ($H_0: \mu_{\Delta} \le 0$) if paired $t$-test $p < 0.001$ with positive mean difference $\Delta = \text{TP}_{\text{H2}} - \text{TP}_{\text{H1}} > 0$.

### Hypothesis $H_{\text{scale}, 1b}$ (Branch B Structural Metric Divergence — Confirmatory Replication)
- **Prediction:** Across the 1,000-table sample, the heuristic merge pipeline (H1) will score higher on global tree-edit distance structure (TEDS-S) and topology similarity (GriTS-Top) than Exact Set Packing (H2), replicating the baseline divergence observed on 100 tables (H1 TEDS-S $0.7318$ vs. H2 $0.6725$; H1 GriTS-Top $0.7908$ vs. H2 $0.7605$), despite H2 dominating cell recall.
- **Mechanism:** Upstream split models over-segment table rows; heuristic separator pruning deletes empty cuts, generating a shallow HTML tree that closely matches ground-truth tree depth. Exact optimization preserves granular physical cells on over-split rows, which tree-edit metrics penalize.
- **Decision Rule:** Confirm Branch B confirmatory replication if mean TEDS-S difference $\Delta_{\text{TEDS}} = \text{TEDS}_{\text{H1}} - \text{TEDS}_{\text{H2}} > 0$ with $p < 0.01$ (paired $t$-test).

### Hypothesis $H_{\text{scale}, 2}$ (TableFormer De-Snapped Statistical Equivalence & Factorial Replication)
- **Prediction:** In the `docling-models v2.1.0` implementation of TableFormer, Exact Maximum Weight Independent Set (MWIS) selection on continuous regression boxes will achieve statistical equivalence with the raw continuous regressor within an equivalence margin of $\pm 2.0$ TPs at $\alpha=0.05$ under Two One-Sided Tests (TOST). Furthermore, MWIS under realistic OCR word-boxes will achieve a statistically significant cell recall rescue over honest de-snapped heuristic post-processing ($\Delta > 0, p < 0.001$, scaling the $+281$ TP rescue).
- **Factorial Replication:** Snapping to OCR word boxes will trigger an empirical drop of $\ge 15\%$ in TPs relative to raw continuous regression (replicating $5,204 \rightarrow 3,902$ TPs, $-25.0\%$), while disabling snapping on OCR word-boxes will stabilize performance at $\approx 85\% - 90\%$ recall (replicating $4,921$ TPs), confirming that snapping to sub-cell OCR words causes $\ge 70\%$ of the collapse from raw to OCR-snapped.
- **Decision Rule:** Reject null ($H_0: \Delta_{\text{TOST}} \ge 2.0$) if TOST $p < 0.05$ and MWIS vs. Heuristic-OCR rescue $p < 0.001$.

### Hypothesis $H_{\text{scale}, 3}$ (Normalized Greedy Selection Identity)
- **Prediction:** Under footprint-normalized candidate weights ($w_i / |\text{tiles}_i|$), greedy selection will achieve a mean per-table Jaccard similarity of $\ge 0.98$ with the Exact Set Packing ILP solution, with $\ge 85\%$ of tables exhibiting identical cell partitions (Jaccard = 1.0000), replicating the baseline finding (1,835 vs. 1,836 TPs; mean Jaccard = 0.9962; 93/100 identical). Under unnormalized coverage sums, greedy selection will collapse to $< 50\%$ of ILP cell recall (replicating $504$ vs. $1,836$ TPs).
- **Decision Rule:** Confirm that objective formulation, rather than solver capability, is the primary determinant of post-processing selection quality.

---

## 6. Statistical Testing Protocol & Multiplicity Control

- **Confirmatory Family:** Paired tests across 1,000 tables will use two-tailed paired $t$-tests, Wilcoxon signed-rank tests, and 10,000-resample bootstrap 95% confidence intervals.
- **Multiplicity Correction:** Holm-Bonferroni step-down procedure applied to maintain Family-Wise Error Rate (FWER) at $\alpha = 0.05$ across the four primary hypothesis families ($H_{\text{scale}, 1a}, H_{\text{scale}, 1b}, H_{\text{scale}, 2}, H_{\text{scale}, 3}$).
- **Equivalence Bounds:** Two One-Sided Tests (TOST) parameterized with an equivalence margin of $\pm 2.0$ TPs ($\approx 3.5\%$ of mean table cell count).

---

## 7. Software Environment & Codebase Freezing

- **Git Commit Hash:** `cdcda075c37aaaf1499be92667ea7aeac9a750c3`
- **MILP Solver:** HiGHS via SciPy 1.15.1 (`scipy.optimize.milp`).
- **Deep Learning Framework:** PyTorch 2.5.1+cu121.
- **Frozen Evaluator Scripts & Checksums:**
  - `canonical_greedy_suite_results.csv`: `1d54969b759a9bfcc646a25e8a616b007d2e10bb6952253a8f428cf21169ba07`
  - `canonical_greedy_suite_summary.json`: `f9a85caff9d81588b0c559f51c57cacc8af290ca68f01bf4ad2e608acb94313e`
  - `metric_suite_results.csv`: `0a5c8d3a8244131deeb3e5871981339bbf1d23d9cf0348617e15bb59d9cb083c`
  - `metric_suite_summary.json`: `56c727424a627d3347a2f3bc75527ee7dd5f55df9d907fa26630e923b2115620`
  - `tatr_pubtables1m_consistency_report.csv`: `ccf7908d2a94f4b652e7202974ac5543fd998caa8b012d42636939b5af4da31e`
  - `lore_threshold_sweep_results.csv`: `4bd31fec232b7c7b6d2cce48107dd192ed997345c6ea672060d52e772530460f`
  - `scratch/lore_matcher_comparison_results.json`: `b5ebe3c25f948db06722eccbd303bda8c5293e58da5e858b01cbcc6ecefdb903`
  - `scratch/tableformer_2x2_factorial_results.csv`: `ae76718c9f629302ee0ed9f6e607c5751a2d719080cebec53dff384fbd367b6e`
  - `scratch/tableformer_mwis_factorial_results.csv`: `195f3935d94aa48adbcc62a6da1ceb2677d135620b86452de1fdaa630500c118`
