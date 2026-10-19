# TableFormer H2 Evaluation: Conflict Definition Tolerance, Statistical Parity & Optimization Limits

**Date:** September 26, 2026  
**Focus:** Hypothesis 2 (H2) Validation on Paradigm 1 (TableFormer-fast) across 100 PubTabNet Tables  
**Status:** Full 100-Table Experimental Suite, Fine Sweep, & Coordinate-Level Audit Complete  
**Artifacts Produced:**
- [`tableformer_h2_metrics.csv`](file:///d:/RRC/major_project_csvtu/tableformer_h2_metrics.csv): Per-table True Positive metrics across Raw, Heuristic, and MWIS configurations.
- [`scratch/tableformer_tolerance_sweep_results.json`](file:///d:/RRC/major_project_csvtu/scratch/tableformer_tolerance_sweep_results.json): Initial coarse empirical sweep over 10 conflict definitions.
- [`scratch/analyze_tableformer_fine_sweep.py`](file:///d:/RRC/major_project_csvtu/scratch/analyze_tableformer_fine_sweep.py): Fine-grained sweep driver and statistical inference test.

---

## 1. Executive Summary

In Hypothesis 1 (H1), we showed that TableFormer's heuristic `MatchingPostProcessor` is load-bearing: without it, 57% of tables exhibit overlapping raw bounding boxes. However, heuristic deduplication also over-prunes valid cells on ordinary tables and collapses recall to 17.6% on extreme autoregressive tail predictions.

In Hypothesis 2 (H2), we tested whether replacing heuristic deduplication with an exact Maximum Weight Independent Set (MWIS) solver over the raw box graph—using the content-tightness objective ($\text{Token Area} / \text{Box Area}$)—could resolve overlaps without destroying valid cells.

This evaluation yielded two distinct, carefully delineated conclusions:

1. **The Methodological Finding (The 190-Cell Collapse):** The apparent catastrophic loss of 190 True Positives under our initial implementation (5,014 vs. 5,204 TPs) **does not reflect an architectural incompatibility between continuous regression and exact optimization**. Rather, it was an artifact of an overly brittle conflict definition (`intersection > 0`) that treated harmless sub-pixel boundary bleed ($0.1$–$0.5\text{ px}$) between legitimate adjacent cells as hard discrete conflicts. Gating conflict edges with substantive tolerances (matching the criterion applied to TATR) eliminates **98.9% to 99.5% of the gap**.
2. **The Optimization Finding (Statistical Parity with Raw):** While a tolerant conflict definition completely rescues valid cells from false conflict pruning, exact MWIS achieves **statistical parity with the raw regressor, but cannot net-create True Positives** (5,203 vs. 5,204 TPs, paired $t = -1.000$, $p = 0.320$, 95% bootstrap CI $[-0.0300, 0.0000]$). Across a fine sweep of 28 distinct threshold settings, MWIS never achieved a net win over raw regression ($Wins = 0$ everywhere). On ordinary tables, TableFormer's non-catastrophic errors are continuous coordinate drift rather than discrete selection conflicts. MWIS functions as an effective guardian against catastrophic hallucination loops (e.g. `_629558`), but is bounded by the continuous coordinate precision of the underlying regressor.

---

## 2. Benchmark Baselines ($N = 100$ Tables, 5,646 Ground Truth Cells)

| Evaluation State | Total Predicted Cells | TP ($\text{IoU} \ge 0.50$) | Recall | Precision | $\text{F}_1$ Score | vs. Raw TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Raw Regressed Boxes** | 6,009 | 5,204 | 92.17% | 86.60% | 89.30% | — |
| **2a. Honest De-Snapped Heuristic Repair** | 5,502 | 5,082 | 89.99% | 92.37% | 91.16% | -122 |
| **2b. Circular Snapped Pipeline (`MatchingPostProcessor`)** | 5,502 | 5,488† | 97.20% | 99.75% | 98.46% | +284 |
| **3. Strict MWIS (`intersection > 0`)** | 5,348 | 5,014 | 88.81% | 93.75% | 91.21% | -190 |
| **4. Tolerant MWIS (10% bleed tolerance)** | 5,901 | 5,174 | 91.64% | 87.68% | 89.62% | -30 |
| **5. Tolerant MWIS (TATR-parity, 50% overlap)** | 5,992 | 5,202 | 92.14% | 86.82% | 89.40% | -2 |
| **6. Tolerant MWIS (Standard NMS, $\text{IoU} > 0.50$)** | 6,003 | 5,203 | 92.15% | 86.67% | 89.33% | -1 |

† *Note on Heuristic Snapping Circularity:* The 5,488 heuristic TP count is artificially inflated by +406 over honest de-snapped post-processing (and +284 over raw) purely due to evaluation circularity: `MatchingPostProcessor` forcibly overwrites sloppy regressed coordinates with ground-truth OCR token boundaries in step 8a (`_align_table_cells_to_pdf`) prior to IoU evaluation. Disabling snapping reveals that the native post-processor actively destroys 122 valid cells ($5,204 \rightarrow 5,082$), which exact MWIS rescues (+121 cells, achieving 5,203 TPs).

---

## 3. Fine-Grained Conflict Tolerance Sweep

To test whether there is any crossover point where MWIS exceeds the raw regressor, we swept 14 IoU gating thresholds and 14 proportional overlap thresholds across all 100 benchmark tables ($5,646$ ground truth cells):

### 3.1 IoU Gating Sweep

| IoU Threshold | MWIS TP | vs. Raw TP (5,204) | Table Wins | Table Ties | Table Losses | Paired $t$-test ($p$-value) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.05** | 5,177 | -27 | 0 | 82 | 18 | $t = -3.97, p = 0.0001$ |
| **0.10** | 5,196 | -8 | 0 | 92 | 8 | $t = -2.93, p = 0.0042$ |
| **0.15** | 5,199 | -5 | 0 | 95 | 5 | $t = -2.28, p = 0.0246$ |
| **0.20** | 5,200 | -4 | 0 | 96 | 4 | $t = -2.03, p = 0.0449$ |
| **0.25** | 5,201 | -3 | 0 | 97 | 3 | $t = -1.75, p = 0.0832$ *(Not significant)* |
| **0.30** | 5,202 | -2 | 0 | 98 | 2 | $t = -1.42, p = 0.1583$ *(Not significant)* |
| **0.35** | 5,203 | -1 | 0 | 99 | 1 | $t = -1.00, p = 0.3197$ *(Not significant)* |
| **0.40** | 5,203 | -1 | 0 | 99 | 1 | $t = -1.00, p = 0.3197$ *(Not significant)* |
| **0.45** | 5,203 | -1 | 0 | 99 | 1 | $t = -1.00, p = 0.3197$ *(Not significant)* |
| **0.50** | 5,203 | -1 | 0 | 99 | 1 | $t = -1.00, p = 0.3197$ *(Not significant)* |
| **0.55** | 5,203 | -1 | 0 | 99 | 1 | $t = -1.00, p = 0.3197$ *(Not significant)* |
| **0.60** | 5,204 | +0 | 0 | 100 | 0 | Identical to Raw |
| **0.70** | 5,204 | +0 | 0 | 100 | 0 | Identical to Raw |
| **0.80** | 5,204 | +0 | 0 | 100 | 0 | Identical to Raw |

### 3.2 Proportional Overlap Sweep ($\text{Intersection} / \min(\text{Area}_1, \text{Area}_2)$)

| Overlap Ratio | MWIS TP | vs. Raw TP (5,204) | Table Wins | Table Ties | Table Losses | Paired $t$-test ($p$-value) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.05** | 5,133 | -71 | 0 | 72 | 28 | $t = -4.17, p = 0.0001$ |
| **0.10** | 5,174 | -30 | 0 | 81 | 19 | $t = -4.02, p = 0.0001$ |
| **0.15** | 5,186 | -18 | 0 | 85 | 15 | $t = -3.75, p = 0.0003$ |
| **0.20** | 5,193 | -11 | 0 | 89 | 11 | $t = -3.50, p = 0.0007$ |
| **0.25** | 5,196 | -8 | 0 | 92 | 8 | $t = -2.93, p = 0.0042$ |
| **0.30** | 5,200 | -4 | 0 | 96 | 4 | $t = -2.03, p = 0.0449$ |
| **0.35** | 5,200 | -4 | 0 | 96 | 4 | $t = -2.03, p = 0.0449$ |
| **0.40** | 5,200 | -4 | 0 | 96 | 4 | $t = -2.03, p = 0.0449$ |
| **0.45** | 5,201 | -3 | 0 | 97 | 3 | $t = -1.75, p = 0.0832$ *(Not significant)* |
| **0.50 (TATR)** | 5,202 | -2 | 0 | 98 | 2 | $t = -1.42, p = 0.1583$ *(Not significant)* |
| **0.55–0.80** | 5,203 | -1 | 0 | 99 | 1 | $t = -1.00, p = 0.3197$ *(Not significant)* |

### 3.3 Statistical Significance of the Near-Parity Result
At standard NMS tolerance ($\text{IoU} > 0.50$):
- **Raw TPs:** 5,204
- **MWIS TPs:** 5,203 ($\Delta = -1$)
- **Per-table distribution:** **0 Wins, 99 Ties, 1 Loss**
- **Paired $t$-test:** $t = -1.0000, p = 0.3197$
- **Wilcoxon signed-rank test:** $W = 0.0000, p = 0.3173$
- **10,000-Resample Bootstrap 95% CI on Mean Difference:** $[-0.0300, 0.0000]$

**Conclusion:** The single-cell difference between MWIS and Raw across 100 tables is **statistically indistinguishable from zero ($p = 0.320$)**. It is not a statistically detectable penalty, nor is there any crossover threshold where MWIS achieves a net positive gain ($Wins = 0$ across all 28 tested threshold settings).

---

## 4. Coordinate-Level Worked Traces

To maintain methodological consistency with the other three paradigms, we examine the literal bounding-box coordinates for both the strict failure mode and the single tolerant divergence.

### 4.1 Strict Mode Failure Trace: Sub-Pixel Regression Bleed (`pubtabnet_tok_670433`)
In table `pubtabnet_tok_670433`, TableFormer predicted two genuinely adjacent vertical cells:
- **Upper Box:** `[462.18, 68.32, 495.25, 79.06]`
- **Lower Box:** `[462.21, 78.82, 495.30, 89.20]`

The lower box's top boundary ($y = 78.82$) bled $0.24$ pixels above the upper box's bottom boundary ($y = 79.06$).
- Under `intersection > 0`, the solver computed an overlap area of $0.24 \times 33.07 = 7.94\text{ px}^2$.
- This triggered an artificial conflict edge between the two cells.
- The upper box had content-tightness $0.9379$; the lower box had $0.8378$.
- The solver selected the upper box and deleted the lower box, destroying a perfectly valid cell detection over $0.24$ pixels of continuous regression bleed.

### 4.2 Tolerant Mode Trace: The Single Divergence at $\text{IoU} > 0.50$ (`pubtabnet_tok_713774`)
Across all 100 tables at $\text{IoU} > 0.50$, exactly one table diverged from raw: `pubtabnet_tok_713774` (Raw: 13 TPs, MWIS: 12 TPs).

In this table, Ground Truth token 11 (`"Balague, 1995[57]..."`) has ground-truth bbox:
$$\text{GT}_{11} = [205, 128, 304, 234] \quad (\text{Width}=99\text{ px}, \text{Height}=106\text{ px})$$

TableFormer emitted two competing vertical prediction boxes over this text:
- **Box #10:** `[215.34, 121.52, 300.69, 170.95]` ($\text{Width}=85.35\text{ px}, \text{Height}=49.43\text{ px}$)
- **Box #13:** `[213.17, 126.30, 312.76, 190.28]` ($\text{Width}=99.59\text{ px}, \text{Height}=63.98\text{ px}$)

Both boxes under-predicted the vertical height of the multi-line reference. Evaluating their overlap and objective weights:
1. **Mutual Conflict:** $\text{IoU}(\text{Box}_{10}, \text{Box}_{13}) = \mathbf{0.5620} > 0.50$. The two boxes are in direct conflict.
2. **Content-Tightness Objective:**
   - Box #10 covers only the top portion of the text, yielding tightly packed token area: $\text{Tightness} = \mathbf{0.9828}$.
   - Box #13 extends further down ($y=190.28$), capturing more characters but including slightly more lateral white space: $\text{Tightness} = \mathbf{0.9164}$.
3. **Solver Decision:** Because $0.9828 > 0.9164$, MWIS selects Box #10 and prunes Box #13.
4. **Impact on Evaluation Metric:**
   - $\text{IoU}(\text{Box}_{10}, \text{GT}_{11}) = 0.3318$ (Fails the $\text{IoU} \ge 0.50$ threshold).
   - $\text{IoU}(\text{Box}_{13}, \text{GT}_{11}) = \mathbf{0.5047}$ (Barely squeaks over the threshold by $0.0047$).

In the raw predictions, Box #13 was present, scoring 1 TP. Because MWIS preferred the tighter, truncated Box #10 over the looser box that barely crossed the arbitrary $0.50$ threshold, the table's TP count shifted from 13 to 12.

> [!NOTE]
> **Mechanistic Distinction:** MWIS did not fail here because the solver became confused by noise or suffered algorithmic breakdown. Rather, the optimizer faithfully maximized its content-tightness proxy ($0.9828 > 0.9164$), but the external evaluation metric ($\text{IoU} \ge 0.50$ vs. GT) happened to fall on the opposite side of a knife-edge threshold ($0.5047$ vs $0.3318$). This is an unresolvable surrogate-alignment gap—an inference-time proxy cannot anticipate ground-truth boundary decisions without oracle access in advance—distinct from TATR's sub-pixel geometric variance.

## 5. Architectural Synthesis: Why Optimization Cannot Exceed Raw Regression

This experiment clarifies why exact combinatorial optimization succeeds dramatically in one paradigm but reaches statistical parity in another:

1. **SPLERGE (Discrete Partition Assembly — $+632$ TPs):**
   - The input space consisted of fractured 1D grid tiles that physically split multi-line words.
   - Set Packing operated over a candidate pool of merged tile unions ($\mathcal{C}_{\text{merge}}$), using token containment to actively *synthesize* missing multi-tile spans. Optimization was generative and structural.
2. **TableFormer (Direct Coordinate Regression — Parity / $\Delta = -1$ TP):**
   - The model directly outputs one bounding box query per cell. On non-catastrophic tables, there are no fractured tiles to assemble; each cell already has exactly one predicted box.
   - The errors are continuous regression imprecisions (e.g., an edge regressed 4 pixels too narrow, yielding IoU 0.48 instead of 0.50).
   - Maximum Weight Independent Set is a **discrete selection operator**, not a **continuous coordinate refiner**. It can select or reject candidate boxes, but cannot nudge continuous boundaries into metric compliance.
   - On catastrophic hallucination tails (`pubtabnet_tok_629558`), where autoregressive decoding looped and produced 345 boxes for a 34-cell table, MWIS acts as an effective filter, pruning hallucinated clutter and boosting TPs from 6 to 13.
   - But across the typical corpus, discrete selection cannot create net new True Positives out of continuous coordinate regression error.
