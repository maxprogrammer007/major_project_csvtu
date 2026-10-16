# H2 Evaluation: Principled Constrained Optimization Replaces Destructive Heuristic Repair

**Date:** September 20, 2026  
**Focus:** Hypothesis 2 (H2) Validation on SPLERGE across 100 PubTabNet Tables  
**Status:** Full 100-Table Experimental Suite Complete  
**Artifacts Produced:**  
- [`splerge_h2_per_table_metrics.csv`](file:///d:/RRC/major_project_csvtu/splerge_h2_per_table_metrics.csv): Complete per-table metrics across Raw, Heuristic, and H2 configurations.
- [`splerge_h2_audit_summary.json`](file:///d:/RRC/major_project_csvtu/splerge_h2_audit_summary.json): Aggregate summary statistics, recovery rates, and solver runtime percentiles.
- [`optimize_splerge_consistency.py`](file:///d:/RRC/major_project_csvtu/optimize_splerge_consistency.py): Automated evaluation driver.

---

## 1. Executive Summary

In Hypothesis 1 (H1), we established across four major Table Structure Recognition (TSR) architectural paradigms that post-hoc heuristic repair mechanisms (e.g. TableFormer's pruning, TATR's row NMS, SPLERGE's `row_merging`) actively destroy valid detections. SPLERGE was identified as the purest manifestation of this phenomenon: an unconditional 1-pixel straddle rule in `merge.py` eliminated 634 true positive cells (34.59% loss), degrading 72/100 tables and improving 0.

In Hypothesis 2 (H2), we replace this destructive repair step with an exact, inference-time **Maximum Weight Set Packing** formulation solved via Integer Linear Programming (ILP).

> [!IMPORTANT]
> **Evaluation Disclosure & "Stress-Test" Framing:** During evaluation, we discovered that the open-source SPLERGE checkpoint (`split_model.pth`) significantly underperforms published benchmarks (achieving only 32% raw recall, compared to 97%+ in the literature). We verified via codebase audit that this is not a preprocessing mismatch, but rather because the checkpoint is a defective, custom Transformer-based rewrite rather than the official FCN architecture.
> 
> Rather than discard the data, we report these results as a highly informative **stress-test of repair-mechanism behavior under degraded input**. When a model struggles and produces highly fragmented predictions (as models do out-of-domain), heuristics collapse catastrophically. The H2 ILP formulation, by contrast, provides a mathematical safety guarantee that holds exactly as well on garbage predictions as on perfect ones.

### Headline Results ($N = 100$ PubTabNet Tables, 5,646 Ground Truth Cells)

| Configuration | Total Predicted Cells | Internal Overlaps | TP ($\text{IoU} \ge 0.25$) | Recall ($\text{IoU} \ge 0.25$) | Precision ($\text{IoU} \ge 0.25$) | $\text{F}_1$ Score | TP ($\text{IoU} \ge 0.50$) | Recall ($\text{IoU} \ge 0.50$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Raw Split Grid** | 7,069 | 0 (0.0%) | 1,833 | 32.47% | 25.93% | 28.83% | 264 | 4.68% |
| **2. Heuristic Merge (SPLERGE)** | 4,333 | 0 (0.0%) | 1,203 | 21.31% | 27.76% | 24.11% | 144 | 2.55% |
| **3. H2 Set Packing (Ours)** | **7,005** | **0 (0.0%)** | **1,836** | **32.52%** | **26.21%** | **29.03%** | **274** | **4.85%** |

- **The True Value of Merging (H2 vs Raw):** When evaluated against the "do nothing" Raw Split Grid baseline, the exact ILP merge formulation yields a statistically insignificant net gain of only +3 TPs (4 wins, 94 ties, 2 losses; $p = 0.320$, paired t-test). This is a major structural finding: the "merge" step of the split-and-merge paradigm, even when executed flawlessly without destroying cells, provides virtually no structural value over simply keeping the raw splits.
- **Reversal of the Heuristic's Destruction (The Stress-Test):** While merging provides no generalizable lift over the raw baseline, the H2 ILP formulation successfully reverses the catastrophic damage caused by SPLERGE's published heuristic. H2 recovered **632 out of 634 destroyed True Positives (99.68%)**, generating an asymmetric win record of 74 wins, 26 ties, and 0 losses against the heuristic on a recall basis. (Note: We measured a strong, statistically significant correlation between raw grid fragmentation and TP destruction: Pearson $r = 0.6534$, Spearman $\rho = 0.6530$, $p < 10^{-12}$. This confirms the massive 634-cell magnitude of the heuristic's damage is significantly associated with the defective checkpoint's noisy output, though 57% of the variance remains unexplained by fragmentation alone. Because the 1-pixel straddle rule is an unconditional, checkpoint-independent code logic flaw, we expect this destructive behavior to generalize to properly-trained models on ambiguous multiline tables).
- **Guaranteed Consistency:** Exactly **0.0% internal overlaps** maintained across all 100 tables (0 overlaps in 7,005 active cells).
- **Inference Runtime:** Mean MILP solve time via HiGHS was **98.22 ms** per table (median: **37.68 ms**, 95th percentile: **358.79 ms**).

---

## 2. Formulation: Maximum Weight Set Packing over Grid Tessellations

### 2.1 The Mathematical Model
Let a table image have height $H$ and width $W$. The 1D horizontal and vertical split projections predict row cut coordinates $Y = [y_1, \dots, y_{R-1}]$ and column cut coordinates $X = [x_1, \dots, x_{C-1}]$, partitioning the table into an $R \times C$ grid of elementary rectangular tiles $\mathcal{U} = \{u_{r, c} : 0 \le r < R, 0 \le c < C\}$.

We generate a candidate pool $\mathcal{C} = \mathcal{C}_{\text{elem}} \cup \mathcal{C}_{\text{merge}}$:
1. **Elementary candidates ($\mathcal{C}_{\text{elem}}$):** The $R \times C$ single-tile rectangles.
2. **Merged candidate spans ($\mathcal{C}_{\text{merge}}$):** Multi-row and multi-column candidate rectangles formed by contiguous tile unions ($[r_1, r_2] \times [c_1, c_2]$) up to height 3 and width 2.

We assign a binary decision variable $x_c \in \{0, 1\}$ for each candidate $c \in \mathcal{C}$, indicating whether candidate $c$ is active in the final table layout.

### 2.2 Optimization Problem
$$\max_{x \in \{0, 1\}^{|\mathcal{C}|}} \sum_{c \in \mathcal{C}} w_c x_c$$
$$\text{subject to } \sum_{c \in \mathcal{C} : u \in c} x_c \le 1 \quad \forall u \in \mathcal{U}$$

In matrix form:
$$\max w^T x \quad \text{s.t.} \quad A x \le \mathbf{1}, \quad x \in \{0, 1\}^{|\mathcal{C}|}$$
where $A \in \{0, 1\}^{|\mathcal{U}| \times |\mathcal{C}|}$ is the binary incidence matrix ($A_{u, c} = 1$ iff candidate $c$ covers elementary tile $u$).

Because $A x \le \mathbf{1}$ enforces that no elementary tile $u$ is covered by more than one selected candidate, the resulting layout is **mathematically guaranteed to have 0.0% internal cell-on-cell overlaps**.

### 2.3 Strict Subadditivity Safeguard ($w_c$) and Theoretical Lower Bound
A critical design requirement is preventing the optimizer from silently recreating the heuristic merge defect. If $w_c$ were simply linear in cell area or token count, a candidate merging two separate data cells $c_1$ and $c_2$ would receive equal or greater weight than keeping them distinct.

To prevent false merges, $w_c$ satisfies **strict subadditivity for distinct tokens**:
$$w(c_1) + w(c_2) > w(c_1 \cup c_2) + \epsilon_{\text{penalty}}$$

#### Principled Calibration of $\epsilon_{\text{penalty}} = 1.0$:
- In table structure recognition semantics, merging two distinct single-line data cells collapses two ground-truth matches into at most one (destroying exactly 1 True Positive cell detection). Setting $\epsilon_{\text{penalty}} = 1.0$ exactly offsets the value of 1 full cell.
- This creates an objective margin of $\ge 1.0$, which is $10^6 \times$ above the HiGHS solver's numerical tolerance ($10^{-6}$), preventing floating-point indifference while leaving genuine merges (which earn $+1.5$ rescue bonus) unaffected.

#### The Theoretical Lower-Bound Guarantee (Safe on Degraded Input):
In the Set Packing formulation, selecting all elementary tiles $\mathcal{X}_{\text{elem}} = \{x_c = 1 \, \forall c \in \mathcal{C}_{\text{elem}}, \, x_c = 0 \, \forall c \in \mathcal{C}_{\text{merge}}\}$ is always a feasible point satisfying $A x \le \mathbf{1}$. Because $\mathcal{X}_{\text{elem}}$ corresponds exactly to the Raw Split Grid, the optimal solution is **mathematically lower-bounded in its objective weight**:
$$W_{\text{opt}} \ge W_{\text{raw}}$$
This is a critical property for production systems: **the guarantee holds exactly as well on garbage raw predictions as on good ones**. Even when the underlying model degrades severely on out-of-distribution input (as this stress-test checkpoint did), the ILP formulation is strictly safe and will never perform structurally worse than the raw grid's objective weight.

It is important to note that this is a guarantee on the ILP's internal proxy weight ($W$), not the final ground-truth-verified True Positive count ($TP$). Because $w_c$ is a heuristic signal computed without access to ground truth, the ILP optimizes its proxy, not the true evaluation metric. As shown in Section 3.4, there are rare cases (2/100 tables) where the optimizer correctly maximized the objective, but the heuristic proxy's judgment diverged from ground truth, resulting in $TP_{\text{opt}} < TP_{\text{raw}}$. This is the expected consequence of $w_c$ being an imperfect heuristic proxy (though it proved 98% faithful), not a violation of the feasibility argument.

#### Directional Scoping & Coordinate Reconciliation Notes:
1. **Directional Scoping:** On table `_554187` specifically, vertical merges dominate because row cuts slice multi-line text, whereas column cuts fall in wide inter-column gutters ($x=52, 145$). In the generalized 100-table solver (`optimize_splerge_consistency.py`), horizontal candidate spans (`h_merge_2`) are actively generated to handle column spans and super-headers.
2. **Coordinate Reconciliation on `PCa-004`:** Ground truth from `pubtabnet_tok_554187_iocr.json` unequivocally specifies `PCa-004` with bbox `[4, 47, 35, 57]`. The `[4, 47, 34, 57]` coordinate cited in earlier exploratory notes was an isolated transcription slip from table `pubtabnet_tok_695383_iocr.json` (where token 12 `"HLA-A2"` has bbox `[4, 47, 34, 57]`). All four flagship cells (`PCa-004`, `PCa-005`, `AY341247.1`, `AC104771.4`) are evaluated against their exact ground truth JSON annotations.

---

## 3. Detailed Results & Statistical Analysis

### 3.1 Aggregate Performance Comparison

```
                      Recall@0.25 Comparison
Raw Split Grid:       [=================>                 ] 32.47% (1,833 TPs)
Heuristic Merge:      [===========>                       ] 21.31% (1,203 TPs)  [-634 TPs Destroyed]
H2 Set Packing:       [=================>                 ] 32.52% (1,836 TPs)  [+633 TPs Recovered]
```

At both IoU thresholds, H2 matches or exceeds the Raw Split Grid while completely eliminating the 34.59% recall collapse introduced by SPLERGE’s heuristic merge:
- **IoU $\ge 0.25$:** Heuristic = 21.31% $\rightarrow$ **H2 = 32.52%** ($+11.21$ percentage points, $+633$ net TPs).
- **IoU $\ge 0.50$:** Heuristic = 2.55% $\rightarrow$ **H2 = 4.85%** ($+2.30$ percentage points, $+130$ net TPs).

### 3.2 Recovery Analysis: Reversing Heuristic Destruction
- Out of 634 cells destroyed by the heuristic, H2 recovered **632 cells (99.68%)**.
- The remaining 2 cells occurred in two complex tables (`_586722` and `_711083`) where extreme dense token packing created ambiguous multi-way straddles. We empirically verified that these 2 missing cells **did not exceed the candidate pool dimensional cap** (height $\le 3$, width $\le 2$); rather, this was a genuine formulation ambiguity where the objective function favored conflicting tiles.

### 3.3 Win / Loss / Tie Distribution (H2 vs Heuristic)

To provide a full, honest picture of the trade-off, here is the win/loss distribution on all three primary axes (where a "win" means H2 strictly outperformed the Heuristic):

- **Recall / TP Count:** **74 Wins, 26 Ties, 0 Losses** (A clean, asymmetric sweep).
- **F1 Score:** **67 Wins, 10 Ties, 23 Losses**
- **Precision:** **45 Wins, 9 Ties, 46 Losses**

The heuristic achieves slightly higher precision (27.76% vs 26.21%) strictly by predicting fewer boxes (4,333 vs 7,005). The massive recall recovery in H2 drives the F1 score higher overall, but the win is recall-driven, not a strict dominance on precision.

### 3.4 Inferential Statistics: The True Value of Merging (H2 vs Raw)

Because H2 maximizes a weight sum that is mathematically lower-bounded by the raw grid's feasible point, beating the heuristic (which we know degraded the raw grid empirically) is largely a guaranteed outcome. 

The genuinely uncertain statistical question is **H2 vs. Raw**: Does the ILP merge step provide genuine, generalizable additional value over the Raw Split Grid, given that "do nothing" was already a guaranteed floor? 

Across the 100 evaluation tables (Recall/TP basis, H2 vs Raw):
- **4 Wins** (H2 correctly merged cells to rescue a token)
- **94 Ties** (H2 correctly elected to do nothing, matching Raw exactly)
- **2 Losses** (H2's heuristic weight proxy favored a merge that decreased actual IoU TP count)

- **Paired Samples t-test:** $t = 1.000, p = 0.320$
- **Wilcoxon Signed-Rank Test:** $W = 6.0, p = 0.317$

**Conclusion:** The improvement is **not statistically significant**. This is a major structural finding: the "split-and-merge" paradigm's merge step, even when executed flawlessly with an exact ILP formulation that prevents destruction, offers virtually zero structural value over just keeping the raw splits.

---

## 4. Solver Scalability & Computational Efficiency

All MILP instances were solved using the open-source HiGHS solver via SciPy's sparse interface (`scipy.optimize.milp` with `scipy.sparse.csc_matrix`).

### Runtime Distribution ($N=100$ tables)
- **Mean Solve Time:** $98.22\text{ ms}$
- **Median Solve Time:** $37.68\text{ ms}$
- **Minimum Solve Time:** $3.25\text{ ms}$
- **95th Percentile:** $358.79\text{ ms}$
- **Maximum Solve Time:** $718.55\text{ ms}$ (on the largest table with 400 grid tiles and 1,800 candidates)

**Conclusion:** Exact combinatorial optimization is fast enough for practical inference pipelines, debunking the assumption that table post-processing requires lossy, greedy heuristics for computational tractability.

---

## 5. Summary Table for the Overall Thesis

| Paradigm | Failure Mechanism Identified (H1) | Proposed Principled Formulation (H2) | Quantitative Outcome |
| :--- | :--- | :--- | :--- |
| **SPLERGE** | Destructive 1-pixel straddle merge eliminated 634 TPs (34.59% loss), degraded 72 tables. | Maximum Weight Set Packing ($A x \le \mathbf{1}$) with subadditive token completeness objective. | **632/634 destroyed TPs recovered (99.68%)**; 74 wins, 26 ties, 0 losses; 0.0% overlaps; mean solve time 98.2 ms. |
