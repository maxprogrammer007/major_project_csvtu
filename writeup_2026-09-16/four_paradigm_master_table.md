# The Four-Paradigm Master Table: Heuristic Vulnerability and Optimization Limits

This document synthesizes the empirical findings across all four modern Table Structure Recognition (TSR) architectural paradigms, evaluated on identical 100-table benchmarks (5,646 total Ground Truth cells).

It addresses two core research questions:
1. **H1 (Heuristic Vulnerability):** Do hardcoded heuristic post-processing steps (NMS, merge rules, grid deduplication) actively destroy valid structural information predicted by neural networks?
2. **H2 (Optimization Rescue & Limits):** Can replacing greedy heuristics with global combinatorial optimization (Exact Set Packing / MWIS) rescue destroyed table structures, and under what conditions?

---

## 1. Master Architectural Comparison Across TSR Paradigms

All conditions are evaluated against a **fixed global denominator of 5,646 Ground Truth cells** across the 100 benchmark tables. Both Micro-average (global) and Macro-average (per-table unweighted mean) recalls are reported to ensure complete denominator transparency.

| Paradigm | Architecture Family | Model Tested | Raw Baseline | H1: Heuristic Post-Processing | H2: Exact Optimization Rescue (MWIS / Set Packing) | Statistical Contrast & Mechanism Verdict |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **Paradigm 1: Sequence Generation** | Autoregressive Attention + Queries | TableFormer-fast (`docling-models v2.1.0` on PubTabNet, $N=100$) | **5,204 TPs** <br>Micro Rec: 92.17% <br>Macro Rec: 93.41% <br>F1: 95.89% | **5,082 TPs** (De-snapped GT Heuristic, 89.99%) <br>vs. **4,921 TPs** (De-snapped OCR Heuristic, 87.16%) <br>vs. **5,488 TPs** († Circular Snapped GT, 97.20%) <br>vs. **3,902 TPs** (Realistic OCR-Snapped Control, 69.11%) | **5,202 TPs** (MWIS with Realistic OCR Word-Boxes, 92.14%) <br>vs. **5,203 TPs** (MWIS with GT Tokens, 92.15%) <br>via Exact MWIS (HiGHS, ~30.8 ms/table) | **Statistical Equivalence & Evaluation Circularity (Full 3×2 Factorial).**<br>• *The Snapping Artifact Dissected:* In `docling-models v2.1.0` evaluated on PubTabNet, `MatchingPostProcessor` achieves 5,488 TPs purely by circular coordinate snapping: step 8a (`_align_table_cells_to_pdf`) overwrites predicted continuous regression coordinates directly with Ground Truth cell annotations, generating an artificial **+406 TP inflation**.<br>• *Decisive Control Experiment (Realistic OCR):* When the native snapping pipeline is fed realistic OCR word-boxes instead of GT annotations, performance collapses to **3,902 TPs** (**-1,302 below raw regression** and **-1,019 below honest de-snapped OCR**), proving that snapping to sub-cell word fragments destroys valid cell envelopes.<br>• *Honest De-Snapped Post-Processing:* When snapping is disabled, the deployed heuristic drops from 5,204 down to **5,082 TPs** on GT tokens (-122 cells) and **4,921 TPs** on OCR word-boxes (-283 cells via column deduplication over-pruning).<br>• *Optimization Rescue without Oracle Advantage:* Exact MWIS weighted by realistic OCR word-box density rescues **+281 valid cells** over honest de-snapped OCR post-processing ($4,921 \rightarrow 5,202$, $t = 4.28, p = 4.36 \times 10^{-5}$; 42 wins, 55 ties, 3 losses), while matching raw regression within 2 cells ($5,204 \rightarrow 5,202$). When weighted by GT tokens, MWIS achieves 5,203 TPs—the gap between oracle GT tokens and realistic OCR word boxes in MWIS selection is exactly 1 single cell across 5,646 GT cells.<br>• *TOST Equivalence:* Within an equivalence margin of $\pm 2.0$ TPs ($\pm 3.54\%$ of mean table GT size), Two One-Sided Tests confirm MWIS (OCR) and Raw are statistically equivalent within $\pm 2.0$ TPs at $\alpha=0.05$ ($t_1 = 140.72, t_2 = -143.56, p_{\text{TOST}} = 5.46 \times 10^{-117}$). |
| **Paradigm 2: Detection-Based** | DETR Object Detection | TATR-v1.1-All (PubTables-1M, $N=100$) | **1,244 TPs** (Raw predicted rows, prior to NMS pruning) | **1,237 TPs** (Greedy NMS, threshold 0.50 for rows, 0.25 for cols) | **1,245 TPs** (Exact MWIS, HiGHS solver) | **Topology-Constrained Optimization.**<br>• *Unified Graph Topology (Rows + Columns):* Auditing all 1,867 candidates (1,286 rows + 581 cols) reveals **98.13% isolated singletons** (1,832 / 1,867), 13 isolated conflict pairs ($K_2$, 0.70%, where Greedy $\equiv$ MWIS), and exactly 3 length-3 chains ($P_3$, $A-B-C$, 0.16%).<br>• *The $+1$ TP Mechanism:* On `PMC6127715_table_0`, MWIS selected the two endpoints of a row $P_3$ chain (weights $2+2=4$), beating greedy NMS which selected the center node (weight 3). The single column $P_3$ chain contributed 0 TPs (columns achieved 574 TPs in raw, NMS, and MWIS). Exactness matters only on rare overlapping chains. |
| **Paradigm 3: Relation Learning** | Anchor-free Point & Corner Pairing (CenterNet) | LORE-TSR (PubTabNet, $N=100$, official `ckpt_ptn`) | **1,761 TPs** at default $\tau = 0.35$ <br>Micro Rec: 31.19% <br>Precision: 99.04% <br>FP: 17, FN: 3,885 | **N/A (Dormant Heuristic).** Physical NMS never triggers because raw emission density is too sparse. | **N/A (Upstream Emission Bound).** Post-processing conflict optimization cannot create candidates where neural emissions are absent. | **The Upstream Threshold & Capacity Policy:**<br>• *Threshold Policy Overturn:* Losses are not inherently unrecoverable. Sweeping $\tau$ from $0.35$ down to $0.20$ recovers genuine cells from **1,761 $\rightarrow$ 5,111 TPs** (Recall $31.19\% \rightarrow 90.52\%$) with negligible precision loss ($99.04\% \rightarrow 99.18\%$, FP=42, F1=94.66%).<br>• *The True Ceilings:* Lowering $\tau$ further to $0.01$ recovers only 69 additional cells (5,180 TPs) while triggering a **$269\times$ False Positive explosion** ($17 \rightarrow 4,576$ FPs), collapsing Precision to $53.10\%$. Furthermore, CenterNet's fixed $K=100$ candidate query budget caps recall on dense tables ($>100$ cells). |
| **Paradigm 4: Split-and-Merge** | 1D FPN Projections + Planar Grid Partitioning | SPLERGE (PubTabNet, $N=100$) | **1,833 TPs** <br>Micro Rec: 32.47% <br>Macro Rec: 33.81% <br>Precision: 27.16% | **1,336 TPs** (H1 with Word-Boxes, 23.66% Micro Rec, -27.11%) <br>**1,203 TPs** (H1 with GT tokens, 21.31% Micro Rec, -34.37%) | **1,836 TPs** (H2 ILP with Word-Boxes, 32.52% Micro Rec) <br>Mean Solve: 60.3 ms | **Massive Cell Recovery vs. Structural Metric Divergence (Branch B):**<br>• *The Executed H1 Pipeline:* SPLERGE's published heuristic executes an exact 5-step pipeline: `col_merging` $\rightarrow$ `row_merging` $\rightarrow$ `remove_redundant_seperators(0.2)` $\rightarrow$ `remove_redundant_seperators(0.4)` $\rightarrow$ `merge_header_v2`. Over-pruning in these steps degrades 70% (OCR word-boxes) to 72% (GT tokens) of tables.<br>• *Cell Recall Recovery:* Exact Set Packing rescues $+500$ valid cells over $H1_{\text{Word-Boxes}}$ ($t = 6.66, p = 1.52 \times 10^{-9}$; Wilcoxon $p = 2.23 \times 10^{-13}$; 71 wins, 0 losses, 29 ties).<br>• *Objective Weighting & Greedy Collapse:* Under natural unnormalized token-coverage sums, standard greedy collapses to **504 TPs** (8.93% Micro Rec) because it greedily takes heavy multi-tile merges. However, when weights are area-normalized ($w_i / |\text{tiles}_i|$), greedy recovers to **1,835 TPs** (32.50% Micro Rec), achieving a mean Jaccard selection identity of 0.9962 with Exact ILP.<br>• *The Structural Metric Trade-Off:* While $H2$ dominates cell recall, $H1$ achieves higher TEDS-S ($0.7318$ vs $0.6725$) and GriTS-Top ($0.7908$ vs $0.7605$). Heuristic pruning deletes over-split rows to match shallow tree depths at the expense of discarding 500 genuine cells. |

† *Note on TableFormer Snapping:* In `docling-models v2.1.0`, step 8a (`_align_table_cells_to_pdf`) snaps regressed cell boundaries directly to the Ground Truth token bounding boxes. Disabling this circular snapping drops heuristic performance to 5,082 TPs (below Raw 5,204), while snapping to realistic OCR word boxes collapses performance to 3,902 TPs (-1,302 below raw). The published 5,488 count is an evaluation circularity artifact.

---

## 2. Rigorous Experimentation & Rebuttal Evidentiary Dossier

### Experiment 1a: Canonical Greedy Selection Under Identical Objective & Footprint Normalization
Reviewers questioned whether SPLERGE H2 gains stemmed from the Set Packing objective or exact MILP optimization. We evaluated greedy selection on the **identical candidate pool and token-coverage weights**, verifying via unit testing on all 100 tables that greedy's conflict check used the **identical edge set $E$** ($A^T A > 0 \iff E_{\text{greedy}}$, 100/100 passed). We evaluated both token sources (Oracle GT tokens vs. Realistic Geometric Word-Boxes) and both weight formulations:

| Condition / Selection Method | Token Representation | Objective Weighting | Total TPs (Fixed Denom = 5,646) | Micro Recall | TEDS-S (Structure) | GriTS-Top (Topology) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Exact ILP (HiGHS)** | Geometric Word-Boxes | Unnormalized Sum ($w_i$) | **1,836** | **32.52%** | **0.6725** | **0.7605** |
| **Standard Greedy (Seed 42)** | Geometric Word-Boxes | Unnormalized Sum ($w_i$) | **504** | **8.93%** | **0.4230** | **0.5629** |
| **Standard Greedy (20 Seeds)** | Geometric Word-Boxes | Unnormalized Sum ($w_i$) | **$512.6 \pm 9.6$** [486, 532] | **9.08%** | **0.4258** | **0.5654** |
| **Normalized Greedy (Seed 42)**| Geometric Word-Boxes | Footprint-Normalized ($w_i / \|\text{tiles}_i\|$) | **1,835** | **32.50%** | **0.6725** | **0.7605** |
| **Normalized Greedy (20 Seeds)**| Geometric Word-Boxes| Footprint-Normalized ($w_i / \|\text{tiles}_i\|$) | **$1,834.6 \pm 0.8$** [1834, 1836] | **32.49%** | **0.6725** | **0.7605** |
| **Exact ILP (HiGHS)** | Oracle GT Tokens | Unnormalized Sum ($w_i$) | **1,839** | **32.57%** | **0.6733** | **0.7594** |
| **Standard Greedy (Seed 42)** | Oracle GT Tokens | Unnormalized Sum ($w_i$) | **495** | **8.77%** | **0.4103** | **0.5551** |
| **Standard Greedy (20 Seeds)** | Oracle GT Tokens | Unnormalized Sum ($w_i$) | **$517.8 \pm 9.7$** [499, 537] | **9.17%** | **0.4182** | **0.5598** |
| **Normalized Greedy (Seed 42)**| Oracle GT Tokens | Footprint-Normalized ($w_i / \|\text{tiles}_i\|$) | **1,838** | **32.55%** | **0.6733** | **0.7594** |
| **Normalized Greedy (20 Seeds)**| Oracle GT Tokens | Footprint-Normalized ($w_i / \|\text{tiles}_i\|$) | **$1,838.0 \pm 0.0$** [1838, 1838] | **32.55%** | **0.6733** | **0.7594** |

**Selection Identity & The Mechanism Finding:**
1. **Per-Table Selection Identity:** Comparing cell selections between Exact ILP and Normalized Greedy yields a **mean Jaccard similarity of 0.9962**. In **93 out of 100 tables**, Jaccard similarity is **exactly 1.0000** (identical cell partitions). In the remaining 7 tables, Jaccard is $\ge 0.84$, differing only on 1–2 marginal tie-breaks.
2. **The Mechanism:** Under unnormalized token-coverage sums, greedy selection collapses ($504$ vs. $1,836$ TPs) because multi-tile merge candidates have weights exceeding any single elementary tile, locking out higher-aggregate sums. When weights are area-normalized, greedy selection matches the Exact ILP.
3. **Intellectual Honesty Verdict:** *"Under unnormalized token-coverage weights, greedy selection collapses (504 vs 1,836 TPs) and exact optimization is necessary; under normalized weights, greedy matches the ILP (1,835 vs 1,836 TPs, mean Jaccard = 0.9962) — the binding constraint is objective design, not solver capability."*

---

### Experiment 1c: TATR Conflict Graph Topology — Rows and Columns Audited
We extended the conflict-graph topology audit across all 1,867 candidates (1,286 rows and 581 columns) across the 100 PubTables-1M tables:

- **Row Candidates (1,286 total):**
  - Isolated Singletons ($K_1$): 1,260 (97.98%).
  - Isolated Conflict Pairs ($K_2$): 10 pairs (0.78%, where Greedy $\equiv$ MWIS).
  - Simple Chains of Length 3 ($P_3$): 2 components (0.16%).
- **Column Candidates (581 total):**
  - Isolated Singletons ($K_1$): 572 (98.45%).
  - Isolated Conflict Pairs ($K_2$): 3 pairs (0.52%).
  - Simple Chains of Length 3 ($P_3$): 1 component (0.17%).
- **Combined Topology:** 98.13% isolated singletons, 13 $K_2$ pairs (0.70%), and 3 $P_3$ chains (0.16%).
- **The $1,244 \rightarrow 1,245$ Delta:** Isolated to table `PMC6127715_table_0`, where MWIS selected the two endpoints of a $P_3$ chain (weights $2+2=4$), beating greedy NMS which selected the center node (weight 3). Exactness matters only on rare overlapping chains.

---

### Experiment 3: Full Multi-Metric Suite Across Token Sources & Decision Rules
All conditions evaluated against the fixed denominator of **5,646 Ground Truth cells**:

| Condition | Token Representation | Total TPs | Micro Recall | Macro Recall | Precision | F1 | TEDS-S (Primary) | GriTS-Top (Primary) | GriTS-Con (Within-Token) | Solve Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw** | None | 1,833 | 32.47% | 33.81% | 27.16% | 28.42% | 0.6680 | 0.7575 | -- | -- |
| **H1-GT** | Oracle GT Tokens | 1,203 | 21.31% | 23.06% | 27.38% | 24.11% | 0.6888 | 0.7633 | 0.6042 | -- |
| **H1-Word-Boxes** | Geometric Word-Boxes | 1,336 | 23.66% | 26.02% | 27.89% | 26.29% | **0.7318** | **0.7908** | -- (omit dummy) | -- |
| **H2-GT** | Oracle GT Tokens | 1,839 | 32.57% | 34.02% | 27.72% | 28.87% | 0.6733 | 0.7594 | **0.6212** | 61.6 ms |
| **H2-Word-Boxes** | Geometric Word-Boxes | **1,836** | **32.52%** | **33.96%** | 27.49% | **28.72%** | 0.6725 | 0.7605 | -- (omit dummy) | 60.3 ms |
| **Greedy (Unnormalized)**| Geometric Word-Boxes | 504 | 8.93% | 10.68% | 13.15% | 10.74% | 0.4230 | 0.5629 | -- (omit dummy) | 0.8 ms |
| **Greedy (Normalized)**  | Geometric Word-Boxes | 1,835 | 32.50% | 33.94% | 27.48% | 28.71% | 0.6725 | 0.7605 | -- (omit dummy) | 0.9 ms |

*Note on Metric Architecture & Word-Boxes:* Morphological token extraction provides word-level bounding box geometry without character transcriptions (dummy tokens). Computing LCS character string matching on dummy tokens yields an artificial $\sim 0.08$ score. In accordance with rigorous scientific practice, GriTS-Con is strictly omitted for geometric word-box conditions and reported strictly on GT conditions with genuine text tokens ($0.6042 \rightarrow 0.6212$, $+0.0170$).

**The Core Scientific Finding (Branch B Confirmed):**
1. **Cell Recall Superiority:** $H2_{\text{Word-Boxes}}$ wins decisively over $H1_{\text{Word-Boxes}}$ ($+500$ net TPs, $+5.00$ cells/table, 95% CI $[3.61, 6.48]$, $t = 6.66, p = 1.52 \times 10^{-9}$; Wilcoxon on cell TPs: $p = 2.23 \times 10^{-13}$; 71 wins, 0 losses, 29 ties). In zero tables does H1 outperform H2 in recall.
2. **The TEDS Metric Pathology:** $H1_{\text{Word-Boxes}}$ scores higher on TEDS-S ($0.7318$ vs $0.6725$, $p = 0.0044$) and GriTS-Top ($0.7908$ vs $0.7605$). This occurs because SplitModel over-segments tables with empty row cuts; $H1$ aggressively deletes rows and separators, simplifying the tree structure to match shallow ground truths at the expense of destroying 500 valid cells. Exact optimization preserves full grid resolution and cell recall, proving that net structural value is metric-dependent.

#### Concrete Case Studies: Demonstrating the TEDS Over-Pruning Pathology
*(Note: Audited recount supersedes earlier preliminary manual annotations).*

| Benchmark Table | Metric Dimension | Ground Truth (GT) | H1 (Heuristic Merge) | H2 (Exact Set Packing ILP) | Divergence / Mechanism |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`pubtabnet_tok_550868`** | **Cell True Positives** <br>Recall <br>Grid Partition <br>HTML Tree Nodes <br>**TEDS-S (Structure)** | **64 cells** <br>100.0% <br>22 rows × 6 cols <br>91 nodes <br>1.0000 | **25 TPs** <br>39.06% <br>22 rows × 3 cols <br>89 nodes <br>**0.9778** | **36 TPs (+11 cells)** <br>**56.25%** <br>33 rows × 4 cols <br>166 nodes <br>**0.5212** | **Heuristic discards 11 valid cells, yet scores +0.4566 higher on TEDS-S!** <br>SplitModel over-split the table into 33 rows. H1 pruned cuts down to 22 rows, matching GT tree height (89 vs 91 nodes). H2 preserved all 36 valid cells on the 33-row grid, but TEDS-S heavily penalized the 11 extra row insertions. |
| **`pubtabnet_tok_597425`** | **Cell True Positives** <br>Recall <br>Grid Partition <br>HTML Tree Nodes <br>**TEDS-S (Structure)** | **102 cells** <br>100.0% <br>27 rows × 7 cols <br>137 nodes <br>1.0000 | **4 TPs (-34 raw TPs)** <br>3.92% <br>26 rows × 4 cols <br>129 nodes <br>**0.9265** | **39 TPs (+35 over H1)** <br>**38.24%** <br>28 rows × 9 cols <br>279 nodes <br>**0.4748** | **Heuristic destroys 34 of 38 raw cells (89.5% loss of detected cells), yet scores 0.9265 TEDS-S!** <br>Raw split captured 38 valid cells. H1's column-pruning collapsed 9 columns down to 4, destroying 34 valid cells (leaving only 4 TPs). Exact ILP preserved the 9 columns, achieving 39 TPs. Yet because H1's pruned 4-column tree matches GT tree dimensions closely (129 nodes vs 137 GT nodes), TEDS-S rewards it with near-perfect similarity (0.9265) while penalizing H2's fine-grained 279-node tree (0.4748). |

---

### Experiment 5: Mechanism Study Stratified by Spanning-Cell Fraction
We examined whether heuristic destruction concentrates specifically on spanning-heavy tables:

| Spanning Stratum | GT Spanning Fraction | Tables ($N$) | Mean GT Cells | Raw TPs | Heuristic TPs | Heuristic Destruction Rate | ILP Rescued TPs | ILP Recovery Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Low Spanning** | $< 10\%$ | 38 | 51.2 | 685 | 502 | **26.71%** | 685 | **100.0%** |
| **Medium Spanning** | $10\% - 25\%$ | 39 | 57.8 | 709 | 471 | **33.57%** | 709 | **100.0%** |
| **High Spanning** | $> 25\%$ | 23 | 62.9 | 439 | 278 | **36.63%** | 437 | **99.57%** |

- **Mechanism Finding:** Heuristic destruction increases monotonically with table complexity, from 26.71% in low-spanning tables to 36.63% in high-spanning tables.
- **Correlation:** Pearson $r = 0.2967$ ($p = 0.00272$), Spearman $\rho = 0.3070$ ($p = 0.00189$). Heuristic destruction concentrates **weakly but significantly** on complex spanning layouts.

---

### Experiment 6b: 100-Table LORE-TSR CenterNet Threshold Sensitivity Sweep
Full evaluation across all 100 benchmark tables (5,646 Ground Truth cells):

| Confidence $\tau$ | Total Preds | TP (Recovered Cells) | FP (False Positive Growth) | FN | Precision | Recall | F1 Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.50** | 1,371 | 1,357 | 14 | 4,289 | 98.98% | 24.03% | 38.68% |
| **0.40** | 1,385 | 1,368 | 17 | 4,278 | 98.77% | 24.23% | 38.91% |
| **0.35 (Default)**| **1,778** | **1,761** | **17** | **3,885** | **99.04%** | **31.19%** | **47.44%** |
| **0.30** | 2,965 | 2,944 | 21 | 2,702 | 99.29% | 52.14% | 68.38% |
| **0.25** | 4,276 | 4,250 | 26 | 1,396 | 99.39% | 75.27% | 85.67% |
| **0.20** | **5,153** | **5,111** | **42** | **535** | **99.18%** | **90.52%** | **94.66%** |
| **0.15** | 5,246 | 5,176 | 70 | 470 | 98.67% | 91.68% | 95.04% |
| **0.10** | 5,313 | 5,180 | 133 | 466 | 97.50% | 91.75% | 94.53% |
| **0.05** | 5,983 | 5,180 | 803 | 466 | 86.58% | 91.75% | 89.09% |
| **0.01** | **9,756** | **5,180** | **4,576** | **466** | **53.10%** | **91.75%** | **67.26%** |

- **Rewriting §4.4:** Low recall in LORE is largely an **upstream threshold policy artifact**: setting $\tau = 0.20$ recovers **90.52% recall** (5,111 TPs) at **99.18% precision** (F1 = 94.66%) without any post-processing intervention.
- **The True Architectural Ceilings:** 
  1. Setting $\tau < 0.10$ produces virtually zero recall gains ($91.68\% \rightarrow 91.75\%$) while triggering a **$269\times$ False Positive surge** ($17 \rightarrow 4,576$ FPs), collapsing precision to 53.10%.
  2. Recall hard-plateaus at $91.75\%$ because CenterNet's fixed $K=100$ candidate query budget caps emission on tables with $>100$ cells.

---

### Experiment 7: Hyperparameter Robustness Sweep & Weight Magnitude Analysis
We tested SPLERGE Set Packing across 9 hyperparameter configurations ($\times 0.5, \times 1.0, \times 2.0$):
- Subadditivity penalty $\epsilon \in \{0.025, 0.05, 0.10\}$
- Rescue bonus $\lambda_{\text{rescue}} \in \{0.75, 1.50, 3.00\}$
- Elementary tile bonus $\lambda_{\text{elem}} \in \{0.005, 0.01, 0.02\}$

**Empirical Result:** Exactly **1,836 True Positives across all 9 configurations** ($\Delta = 0$).

#### Diagnostic: Why the Sweep is Perfectly Invariant (Weight Magnitude Audit)
To understand why the sweep yielded zero variance across all 9 permutations, we audited the actual empirical candidate weight distributions across the 100 tables:
- **Elementary Cell Candidate Weights ($w_{\text{elem}}$):** Mean $= 0.8901$, Median $= 1.0100$, 75th percentile $= 1.0100$, Max $= 17.0100$. (58.5% of elementary weights $\ge 1.0$).
- **Merged Candidate Weights ($w_{\text{merge}}$):** Mean $= 1.7423$, Median $= 1.9500$, 75th percentile $= 1.9500$, Max $= 31.9500$. (51.9% of merge weights $\ge 1.0$).
- **Scale Analysis:** The elementary tile bonus $\lambda_{\text{elem}} \in [0.005, 0.02]$ accounts for only **$\approx 0.6\% - 1.1\%$** of total candidate weights. Candidate selection is overwhelmingly dominated by discrete integer token-coverage steps ($+1.0$ per contained token). The sensitivity sweep was therefore flat because the secondary regularization terms are ~1% of coverage weights, confirming that token-coverage geometry provides the dominant structural signal.

---

## 3. Pinned Dependencies & Reproducibility Environment

- **HiGHS MILP Solver:** SciPy 1.15.1 (`scipy.optimize.milp`, HiGHS binary branch-and-cut)
- **Deep Learning Framework:** PyTorch 2.5.1+cu121, Torchvision 0.20.1
- **Evaluation Libraries:** APTED 1.0.3, lxml 5.3.0, PyMuPDF 1.25.1
- **Solve Times:** SPLERGE ILP mean solve time $= 60.3$ ms/table; TableFormer MWIS $= 4.2$ ms/table.

---

## 4. Canonical Numbers Provenance Appendix

Every reported headline figure in this paper traces directly to a frozen canonical script, CSV artifact, and summary JSON:

| Paradigm / Model | Condition | Metric / Value | Canonical Generation Script | Output CSV Artifact | Summary JSON Artifact |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **TableFormer-fast** | Raw Regressor | 5,204 TPs (92.17% Micro) | `scratch/test_desnapped_tableformer.py` | `scratch/tableformer_snapping_full_100.csv` | `scratch/tableformer_snapping_summary.json` |
| **TableFormer-fast** | De-Snapped Heuristic | 5,082 TPs (89.99% Micro) | `scratch/test_desnapped_tableformer.py` | `scratch/tableformer_snapping_full_100.csv` | `scratch/tableformer_snapping_summary.json` |
| **TableFormer-fast** | Circular Snapped Heuristic | 5,488 TPs (97.20% Micro) | `scratch/test_desnapped_tableformer.py` | `scratch/tableformer_snapping_full_100.csv` | `scratch/tableformer_snapping_summary.json` |
| **TableFormer-fast** | Realistic OCR Snapped Control | 3,902 TPs (69.11% Micro) | `scratch/test_tableformer_ocr_control.py` | `scratch/tableformer_ocr_control_results.csv` | `scratch/tableformer_ocr_control_summary.json` |
| **TableFormer-fast** | Exact MWIS (GT Tokens) | 5,203 TPs (92.15% Micro) | `scratch/run_tableformer_mwis_factorial.py` | `scratch/tableformer_mwis_factorial_results.csv` | `scratch/tableformer_mwis_factorial_summary.json` |
| **TableFormer-fast** | Exact MWIS (OCR Word-Boxes) | 5,202 TPs (92.14% Micro) | `scratch/run_tableformer_mwis_factorial.py` | `scratch/tableformer_mwis_factorial_results.csv` | `scratch/tableformer_mwis_factorial_summary.json` |
| **TATR-v1.1-All** | Raw Predicted Rows | 1,244 TPs | `measure_tatr_consistency.py` | `tatr_pubtables1m_consistency_report.csv` | `tatr_pubtables1m_consistency_report.csv` |
| **TATR-v1.1-All** | Greedy Row NMS | 1,237 TPs | `measure_tatr_consistency.py` | `tatr_pubtables1m_consistency_report.csv` | `tatr_pubtables1m_consistency_report.csv` |
| **TATR-v1.1-All** | Exact MWIS Rows | 1,245 TPs | `measure_tatr_consistency.py` | `tatr_pubtables1m_consistency_report.csv` | `tatr_pubtables1m_consistency_report.csv` |
| **TATR-v1.1-All** | Column Predictions (All) | 574 TPs (Raw, NMS, MWIS) | `measure_tatr_consistency.py` | `tatr_pubtables1m_consistency_report.csv` | `tatr_pubtables1m_consistency_report.csv` |
| **TATR-v1.1-All** | Graph Topology Audit | 98.13% Singletons (1,832/1,867) | `check_tatr_topology.py` | `tatr_pubtables1m_consistency_report.csv` | `scratch/tatr_topology_full_audit.json` |
| **LORE-TSR** | Default Baseline ($\tau=0.35$) | 1,761 TPs (31.19% Micro) | `sweep_lore_thresholds.py` | `scratch/lore_100_tables_sweep_results.csv` | `scratch/lore_threshold_sweep_summary.json` |
| **LORE-TSR** | Optimal Sweep ($\tau=0.20$) | 5,111 TPs (90.52% Micro) | `sweep_lore_thresholds.py` | `scratch/lore_100_tables_sweep_results.csv` | `scratch/lore_threshold_sweep_summary.json` |
| **LORE-TSR** | Ceiling Sweep ($\tau=0.01$) | 5,180 TPs (FP=4,576) | `sweep_lore_thresholds.py` | `scratch/lore_100_tables_sweep_results.csv` | `scratch/lore_threshold_sweep_summary.json` |
| **SPLERGE** | Raw Split Grid | 1,833 TPs (32.47% Micro) | `run_full_metric_suite.py` | `metric_suite_results.csv` | `metric_suite_summary.json` |
| **SPLERGE** | H1 Heuristic (GT Tokens) | 1,203 TPs (21.31% Micro, -34.37%) | `run_full_metric_suite.py` | `metric_suite_results.csv` | `metric_suite_summary.json` |
| **SPLERGE** | H1 Heuristic (Word-Boxes) | 1,336 TPs (23.66% Micro, -27.11%) | `run_full_metric_suite.py` | `metric_suite_results.csv` | `metric_suite_summary.json` |
| **SPLERGE** | Exact ILP (Word-Boxes) | 1,836 TPs (32.52% Micro) | `scratch/run_canonical_greedy_suite.py` | `canonical_greedy_suite_results.csv` | `canonical_greedy_suite_summary.json` |
| **SPLERGE** | Exact ILP (GT Tokens) | 1,839 TPs (32.57% Micro) | `scratch/run_canonical_greedy_suite.py` | `canonical_greedy_suite_results.csv` | `canonical_greedy_suite_summary.json` |
| **SPLERGE** | Standard Greedy (Word-Boxes) | 504 TPs (Seed 42); $512.6 \pm 9.6$ (20 seeds) | `scratch/run_canonical_greedy_suite.py` | `canonical_greedy_suite_results.csv` | `canonical_greedy_suite_summary.json` |
| **SPLERGE** | Normalized Greedy (Word-Boxes) | 1,835 TPs (Seed 42); $1,834.6 \pm 0.8$ (20 seeds) | `scratch/run_canonical_greedy_suite.py` | `canonical_greedy_suite_results.csv` | `canonical_greedy_suite_summary.json` |

### TableFormer Complete 3×2 Factorial Decomposition Table ($N=100$, 5,646 GT Cells)
Scripts: [`scratch/run_tableformer_2x2_factorial.py`](file:///d:/RRC/major_project_csvtu/scratch/run_tableformer_2x2_factorial.py) & [`scratch/run_tableformer_mwis_factorial.py`](file:///d:/RRC/major_project_csvtu/scratch/run_tableformer_mwis_factorial.py)

| Post-Processing Condition | Oracle GT Tokens | Realistic OCR Word-Boxes | Snapping Effect ($\Delta_{\text{snap}}$) |
| :--- | :---: | :---: | :---: |
| **Snapping ENABLED (Heuristic Snap ON)** | **5,488 TPs** (97.20%) | **3,902 TPs** (69.11%) | Token Source Gap: **-1,586 TPs** |
| **Snapping DISABLED (Heuristic Snap OFF)**| **5,082 TPs** (89.99%) | **4,921 TPs** (87.16%) | Token Source Gap: **-161 TPs** |
| **Exact MWIS Optimization (Ours)** | **5,203 TPs** (92.15%) | **5,202 TPs** (92.14%) | Token Source Gap: **-1 TP** |
| **Raw Continuous Regressor Baseline** | **5,204 TPs** (92.17%) | — | — |
| **Net Snapping Impact** | **+406 TPs** (Circular Inflation) | **-1,019 TPs** (Sub-Cell Mutilation) | **Interaction Delta: 1,425 TPs** |

*(Note: Under realistic OCR word boxes, MWIS rescues +281 cells over honest de-snapped heuristic post-processing, $p = 4.36 \times 10^{-5}$, and is equivalent to Raw continuous regression within $\pm 2.0$ TPs at $\alpha=0.05$, $p_{\text{TOST}} = 5.46 \times 10^{-117}$).*

### Frozen Canonical Data Artifact SHA-256 Hashes

| Artifact Path | SHA-256 Checksum |
| :--- | :--- |
| `canonical_greedy_suite_results.csv` | `1d54969b759a9bfcc646a25e8a616b007d2e10bb6952253a8f428cf21169ba07` |
| `canonical_greedy_suite_summary.json` | `f9a85caff9d81588b0c559f51c57cacc8af290ca68f01bf4ad2e608acb94313e` |
| `metric_suite_results.csv` | `0a5c8d3a8244131deeb3e5871981339bbf1d23d9cf0348617e15bb59d9cb083c` |
| `metric_suite_summary.json` | `56c727424a627d3347a2f3bc75527ee7dd5f55df9d907fa26630e923b2115620` |
| `tatr_pubtables1m_consistency_report.csv` | `ccf7908d2a94f4b652e7202974ac5543fd998caa8b012d42636939b5af4da31e` |
| `lore_threshold_sweep_results.csv` | `4bd31fec232b7c7b6d2cce48107dd192ed997345c6ea672060d52e772530460f` |
| `scratch/tableformer_2x2_factorial_results.csv` | `ae76718c9f629302ee0ed9f6e607c5751a2d719080cebec53dff384fbd367b6e` |
| `scratch/tableformer_2x2_factorial_summary.json` | `be8a609cb20ccb9e4c71928b51df3e6f03b3074b6da223cec5796aa61c9d39ea` |
| `scratch/tableformer_mwis_factorial_results.csv` | `195f3935d94aa48adbcc62a6da1ceb2677d135620b86452de1fdaa630500c118` |
| `scratch/tableformer_mwis_factorial_summary.json` | `ec2c9b99c7ba2fe9ddff439374d127d8445314a20c27e080e1de64a2c8087207` |

### Code Provenance Exhibits

#### Exhibit A: Ground-Truth Cells Used as "PDF Text Tokens" (`run_with_tokens.py`, lines 68–99)
```python
def _extract_iocr_tokens(html_str, img_w, img_h):
    """Convert PubTabNet html cell annotations into Docling IOCR token format."""
    data = _parse_html_field(html_str)
    cells = data.get("cells", [])
    tokens = []
    tok_id = 0
    for cell in cells:
        bbox = cell.get("bbox")
        if not bbox or len(bbox) < 4:
            continue  # empty cell — no spatial anchor
        raw_tokens = cell.get("tokens", [])
        text_chars = [t for t in raw_tokens if not (t.startswith("<") and t.endswith(">"))]
        tokens.append({
            "id": tok_id,
            "text": "".join(text_chars).strip() or " ",
            "bbox": list(bbox),   # [x0, y0, x1, y1] — PubTabNet ground-truth cell bbox!
        })
        tok_id += 1
    return tokens, [0, 0, img_w, img_h]
```

#### Exhibit B: Overwriting Predicted Bounding Boxes with PDF Token Coordinates (`matching_post_processor.py`, lines 495–498)
```python
# Create new table cell with aligned bbox
new_table_cell = table_cell.copy()
new_table_cell["bbox"] = list(pdf_cell_bbox)  # Overwritten with Ground Truth cell bbox!
```

### Statistical Reconciliation of Provenance Anomalies
1. **The 1,836 vs. 1,839 GT Count:** In exploratory work, `optimize_splerge_consistency.py` reported 1,836 GT TPs using an early first-come-first-served prediction loop (`for pb in pred_boxes:`). The canonical pipeline (`run_full_metric_suite.py` and `scratch/run_canonical_greedy_suite.py`) employs a global bipartite maximum-IoU matcher (`np.argsort(iou_matrix.ravel())[::-1]`), which resolves ties to yield 1,839 GT TPs and 1,836 Word-Box TPs. Both canonical files are arithmetically unified.
2. **Normalized Greedy (1,835) vs. Raw (1,833):** Under footprint normalization, greedy selection does not simply default to all elementary tiles; it actively selects 56 valid merge candidates with high token density across the benchmark, achieving +2 TPs over the raw split grid. Under identical matchers, Exact ILP selects optimal combinations to achieve 1,836 TPs (+3 over raw).
3. **SPLERGE H2 vs. Raw Statistical Equivalence:** Comparing H2 (1,836) against Raw (1,833) across all 100 tables yields a mean difference of $+0.0300$ cells/table ($SE = 0.0481$, paired $t = 0.6236, p = 0.5343$, Wilcoxon $W = 28.5, p = 0.4888$, bootstrap 95% CI $[-0.0700, +0.1200]$). Under Two One-Sided Tests (TOST) with margin $\pm 2.0$ TPs, equivalence is established at $p_{\text{TOST}} = 3.55 \times 10^{-64}$.
4. **Heuristic Degradation Audit:** On GT tokens, H1 degrades 72/100 tables (ties 28, improves 0). On OCR word-boxes, H1 degrades 70/100 tables (ties 29, improves 1). Exact ILP achieves 71 wins, 29 ties, and 0 losses against H1.
5. **Greedy Suite History:** Early exploratory greedy trials exhibited drift (710.8, 566) due to non-identical candidate caches and unconstrained tie-breaking. All figures are now superseded by the frozen canonical suite (`canonical_greedy_suite_results.csv`), verified across 20 randomized seeds with identical constraint graphs.
6. **LORE Matcher Protocol Verification:** We empirically evaluated official verbatim `pairTab` matching (`AdvancedLiterateMachinery/.../LORE-TSR/src/lib/utils/eval_utils.py`) against canonical global bipartite matching across all 100 tables. In all three conditions ($\tau = 0.35, 0.20, 0.01$), table matching identity and cell assignment identity are **100.00%** (100/100 tables match identically; 0 TPs discrepancy; 1,761/1,761 pairs at $\tau=0.35$, 5,111/5,111 pairs at $\tau=0.20$, and 5,180/5,180 pairs at $\tau=0.01$). CenterNet's non-overlapping peak detection produces zero overlapping candidate competition at $\text{IoU} \ge 0.50$, proving protocol equivalence across the entire corpus (`scratch/lore_matcher_comparison_results.json`).
7. **H2_GT vs. H2_OCR Per-Table Breakdown & Statistical Contrast:** Across all 100 benchmark tables, Exact ILP on GT tokens (1,839 TPs) versus Exact ILP on OCR word-boxes (1,836 TPs) yields an exact net difference of $+3$ TPs. Querying `canonical_greedy_suite_results.csv` yields the exact table-by-table difference breakdown ($\Delta = \text{ILP}_{\text{GT}} - \text{ILP}_{\text{OCR}}$):
   - **Ties:** Exactly 95 of 100 tables are identical cell-for-cell ($\Delta = 0$).
   - **GT Wins:** Exactly 3 tables (`pubtabnet_tok_566196`: $9 - 8 = \mathbf{+1}$; `pubtabnet_tok_723463`: $3 - 2 = \mathbf{+1}$; `pubtabnet_tok_578521`: $48 - 45 = \mathbf{+3}$).
   - **OCR Wins:** Exactly 2 tables (`pubtabnet_tok_642815`: $2 - 3 = \mathbf{-1}$; `pubtabnet_tok_665351`: $14 - 15 = \mathbf{-1}$).
   - **Net Arithmetic:** $(+1) + (+1) + (+3) + (-1) + (-1) = \mathbf{+3\text{ TPs}}$ ($1,839 - 1,836$).
   - **Wilcoxon Rank Verification:** Ranking the absolute differences $\{1, 1, 1, 1, 3\}$, the four $|\Delta| = 1$ differences share average rank $\frac{1+2+3+4}{4} = 2.5$; the single $|\Delta| = 3$ difference receives rank 5. The negative differences (OCR wins: $-1, -1$) sum to $W = 2.5 + 2.5 = \mathbf{5.0}$ (with $R_+ = 2.5 + 2.5 + 5.0 = 10.0$), yielding exact two-sided $p = 0.4795$.
   - **Equivalence Statistics:** Mean difference $= +0.0300$ cells/table ($SE = 0.0361$, paired $t = 0.8308, p = 0.4081$, bootstrap 95% CI $[-0.0400, +0.1000]$). Two One-Sided Tests (TOST) within $\pm 2.0$ TPs margin establish equivalence at $t_1 = 56.21, t_2 = -54.55, p_{\text{TOST}} = 5.55 \times 10^{-76}$. The early exploratory paper's statistic ($p = 0.257$) is formally superseded.

### Artifact Supersession Map (Provenance Tracking)

To ensure full reproducibility and prevent citation of preliminary or intermediate exploratory data files, the table below documents the explicit supersession mapping from earlier experimental runs to the frozen canonical artifacts:

| Superseded Artifact | SHA-256 Checksum | Scope & Reason for Supersession | Authoritative Frozen Successor |
| :--- | :--- | :--- | :--- |
| `scratch/tableformer_snapping_full_100.csv` | `ce56abc1277910c829ac93d6dcac940369de460c7227f2004902aee7dfd3dc4e` | Preliminary 1-column GT snapping trial (Raw 5,204 vs. GT Snap ON 5,488 vs. GT Snap OFF 5,082). Lacked OCR word-box control conditions. | `scratch/tableformer_2x2_factorial_results.csv` (Unified 2×2 Factorial) |
| `scratch/tableformer_ocr_control_results.csv` | `e7fdd75823e0a3f401359d84ca4276a9898a84500f4e656f03a49d3757f46794` | Standalone single-column OCR control run (OCR Snap ON 3,902). Integrated into unified factorial matrix. | `scratch/tableformer_2x2_factorial_results.csv` (Unified 2×2 Factorial) |
| `tableformer_h2_metrics.csv` | `03b1738dd976ac7e0cb2ea9ae74d0fb0d59c0633a331732dc2ea711284718b0a` | Preliminary MWIS trial evaluated exclusively on GT tokens (5,203 TPs). Lacked OCR word-box density weights. | `scratch/tableformer_mwis_factorial_results.csv` (Complete MWIS Factorial across OCR and GT tokens) |
| Early Exploratory Script Outputs (`optimize_splerge_consistency.py`) | *N/A (untracked script)* | Reported H2-GT as 1,836 TPs due to an unconstrained first-come-first-served prediction loop (`for pb in pred_boxes:`). | `canonical_greedy_suite_results.csv` (Canonical bipartite matching yielding 1,839 GT / 1,836 OCR TPs) |
| Early exploratory H2_GT vs H2_OCR $p$-value ($p = 0.257$) | *N/A (pre-canonical draft)* | Derived from exploratory unnormalized split runs without verified bipartite tie-breaking. | Canonical Paired Test ($t = 0.8308, p = 0.4081$; Wilcoxon $W = 5.0, p = 0.4795$; TOST $p = 5.55 \times 10^{-76}$) |