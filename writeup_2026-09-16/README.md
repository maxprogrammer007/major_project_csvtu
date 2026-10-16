# Cross-Paradigm TSR Consistency Study: Phase 2 Progress

**Date:** 2026-09-16  
**Repository:** `d:/RRC/major_project_csvtu`

This directory contains the experimental findings, audits, and consolidated comparison tables for the multi-paradigm Table Structure Recognition (TSR) consistency study.

---

## Documents in this Release

1. **[`tatr_consistency_findings.md`](file:///d:/RRC/major_project_csvtu/writeup_2026-09-16/tatr_consistency_findings.md)**:
   - Full evaluation report on **Paradigm 2: Transformer / DETR (`microsoft/table-transformer`)** on PubTables-1M test set ($N=100$).
   - Raw overlap rate (10.0%), absence of catastrophic collapse (0.0%), and empirical proof of destructive heuristic repair (loss of 7 true rows in dense layouts).
   - Side-by-side contrast against Paradigm 1 (TableFormer).
2. **[`../tatr_pubtables1m_consistency_report.csv`](file:///d:/RRC/major_project_csvtu/tatr_pubtables1m_consistency_report.csv)**:
   - Per-table breakdown of GT rows/cols, raw detections, repaired detections, overlaps, TP/FP/FN, precision, recall, F1, and over-pruned counts.
3. **[`../measure_tatr_consistency.py`](file:///d:/RRC/major_project_csvtu/measure_tatr_consistency.py)**:
   - Self-contained measurement script implementing the 5-step instrumented protocol on TATR.
4. **[`h1_pilot_findings.md`](file:///d:/RRC/major_project_csvtu/writeup_2026-09-16/h1_pilot_findings.md)**:
   - Consolidated master report covering all **Four Paradigms** (TableFormer, TATR, LORE-TSR, SPLERGE), the Four-Paradigm Master Table, and all methodological caveats.
5. **[`../lore_bucketed_audit.json`](file:///d:/RRC/major_project_csvtu/lore_bucketed_audit.json)** & **[`../lore_per_table_metrics.csv`](file:///d:/RRC/major_project_csvtu/lore_per_table_metrics.csv)**:
   - Full 100-table audit for **Paradigm 3: Relation Learning (LORE-TSR)** on PubTabNet.
   - Empirical proof of 31.2% detection recall, statistical independence of $K=100$ cap vs. detection threshold ($p=0.43$), and tracing of the latent `BaseDetector.filter()` alignment defect.
6. **[`../splerge_audit_summary.json`](file:///d:/RRC/major_project_csvtu/splerge_audit_summary.json)** & **[`../splerge_per_table_metrics.csv`](file:///d:/RRC/major_project_csvtu/splerge_per_table_metrics.csv)**:
   - Full 100-table audit for **Paradigm 4: Split-and-Merge (SPLERGE)** on PubTabNet.
   - Empirical proof of 0.0% internal overlaps by planar tessellation design, and gross destructive loss of 34.6% of valid cell detections ($1,833 \rightarrow 1,203$) driven by `row_merging`.
7. **[`../measure_splerge_consistency.py`](file:///d:/RRC/major_project_csvtu/measure_splerge_consistency.py)**:
   - Automated 100-table evaluation driver chaining SplitModel predictions into rule-based cell merge heuristics without oracle leakage.
8. **[`h2_optimization_findings.md`](file:///d:/RRC/major_project_csvtu/writeup_2026-09-16/h2_optimization_findings.md)**:
   - Full evaluation report on **Hypothesis 2 (H2): Constrained Optimization Replaces Destructive Heuristic Repair**.
   - Proof of **99.68% recovery of destroyed True Positives (632/634 recovered)**, 0.0% internal overlaps, 74 wins / 26 ties / 0 losses across 100 tables, and sub-100ms real-time HiGHS solve time.
9. **[`../splerge_h2_per_table_metrics.csv`](file:///d:/RRC/major_project_csvtu/splerge_h2_per_table_metrics.csv)** & **[`../splerge_h2_audit_summary.json`](file:///d:/RRC/major_project_csvtu/splerge_h2_audit_summary.json)**:
   - Full 100-table per-table metrics and summary audit comparing Raw Split Grid, SPLERGE Heuristic Merge, and H2 Set Packing.
10. **[`../optimize_splerge_consistency.py`](file:///d:/RRC/major_project_csvtu/optimize_splerge_consistency.py)**:
    - Automated 100-table optimization driver.
11. **[`tableformer_h2_findings.md`](file:///d:/RRC/major_project_csvtu/writeup_2026-09-16/tableformer_h2_findings.md)**:
    - Full evaluation report on **Paradigm 1: TableFormer H2 Conflict Definition & Optimization Limits**.
    - Proof of 99.5% recovery of the apparent 190-cell gap under continuous conflict tolerance, and analysis of why MWIS asymptotically matches but does not exceed raw regression.
12. **[`four_paradigm_master_table.md`](file:///d:/RRC/major_project_csvtu/writeup_2026-09-16/four_paradigm_master_table.md)**:
    - Consolidated cross-paradigm master synthesis covering H1 and H2 across all four architectural families.

---

## Status of Four-Paradigm Master Plan

| # | Paradigm | Architecture | Model / Checkpoint | Dataset | Status |
|:---:|:---|:---|:---|:---|:---:|
| 1 | Sequence Generation | Autoregressive OTSL | TableFormer-fast | PubTabNet | **Completed** |
| 2 | Transformer / DETR | Set prediction / Object queries | TATR-v1.1-All | PubTables-1M | **Completed** |
| 3 | Relation Learning | CenterNet / Logical relation | LORE-TSR (`ckpt_ptn`) | PubTabNet | **Completed** |
| 4 | Split-and-Merge | Semantic segmentation / 1D FPN | SPLERGE | PubTabNet | **Completed** |

---

## Four-Paradigm Master Comparison

| Paradigm | Architecture Family | Model Tested | Raw Geometric Overlaps | Catastrophic Collapse | Primary Architectural Bottleneck | Post-Processing Repair Effect |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **Paradigm 1: Sequence Generation** | Autoregressive Attention + Queries | TableFormer-fast | **57.0%** (57/100) | **5.0%** (5/100) | Spatial uncoupling between grammar token decoder and regression queries | **Destructive heuristic pruning:** drops genuine cells on ordinary tables; collapses recall to 17.6% on extreme tails. |
| **Paradigm 2: Detection-Based** | Two-Stage Object Detection | TATR (Faster R-CNN) | **10.0%** (10/100) | **0.0%** (0/100) | Anchor proposal density in compact table line layouts | **Destructive row NMS:** over-prunes adjacent parallel rows in tight vertical spacing. |
| **Paradigm 3: Relation Learning** | Anchor-free Point & Corner Pairing | LORE-TSR | **< 0.1%** (0/10,000 NMS) | **0.0%** (0/100) | Dual bottleneck: hardcoded $K=100$ capacity cap (23.5% loss on dense) + detection confidence dropping (68.5% loss on standard) | **Latent implementation defect:** `BaseDetector.filter()` index shift (dormant on benchmark due to 0 NMS triggers). |
| **Paradigm 4: Split-and-Merge** | 1D FPN Projections + Planar Grid Partitioning | SPLERGE | **0.0%** (0/100) | **0.0%** (0/100) | Boundary jitter on multiline cells triggering false span rules | **Pure destructive repair:** 34.6% of all true positive cells eliminated ($1,833 \rightarrow 1,203$); 72/100 tables degraded, 0 improved. |

---

### Important Comparability Notes & Methodological Caveats

1. **Defective Open-Source Checkpoint (Stress-Test Framing for SPLERGE):**
   - The open-source SPLERGE checkpoint (`split_model.pth`) tested here significantly underperforms published SPLERGE capabilities. We verified that the checkpoint is a custom, unofficial Transformer-based rewrite rather than the FCN described in the paper, yielding a shattered, noisy raw grid (32% recall @ IoU 0.25).
   - Rather than discard the data, we retain these numbers as a highly informative **stress-test of repair-mechanism behavior under degraded input**: proving that when a model struggles out-of-domain and produces fragmented predictions, heuristics collapse catastrophically, while H2's ILP repair remains mathematically bounded and strictly safe.
   - We measured a strong, statistically significant correlation (Pearson $r = 0.6534$, Spearman $\rho = 0.6530$, $p < 10^{-12}$) between raw grid fragmentation and TP destruction. This confirms that the massive 634-cell *magnitude* of the heuristic damage is significantly associated with this defective checkpoint's fragmented output, though a substantial share of the effect (57% of variance) remains unexplained by fragmentation alone. Because the underlying unconditional 1-pixel straddle flaw in `merge.py`'s logic is checkpoint-independent, we expect this destructive behavior to generalize to properly-trained models on ambiguous multiline tables.
2. **Grid Envelopes vs. Content-Tight Bounding Boxes (Absolute Recall Non-Comparability):**
   - TableFormer, TATR, and LORE-TSR regress **tight bounding boxes hugging cell text**.
   - SPLERGE predicts **2D planar grid envelopes** that tile the entire table canvas (full row-height $\times$ column-width rectangles including inter-word spacing and gutter margins).
   - Under standard Intersection-over-Union matching against tight ground-truth word annotations, a grid envelope containing a short word or number mechanically caps $\text{IoU} \le \frac{\text{Word Area}}{\text{Cell Area}}$, scoring 0 under IoU thresholds even when text containment is 100%.
   - **Empirical Containment Audit:** Auditing all 5,646 ground-truth tokens across all 100 tables revealed that **97.57% of tokens (5,509 / 5,646)** have $\ge 80\%$ of their area geometrically enclosed within a predicted raw grid cell (**99.70%** have $\ge 50\%$ containment). Thus, the SplitModel reliably captures cell contents; the 32.47% ($\text{IoU} \ge 0.25$) and 4.68% ($\text{IoU} \ge 0.50$) recall metrics reflect the envelope-vs-tight representation disparity. **SPLERGE's absolute recall numbers should not be read directly against the tight-box recall numbers of Paradigms 1–3.**
3. **Threshold Shift ($\text{IoU} \ge 0.25$ vs. $\text{IoU} \ge 0.50$):**
   - Due to the grid-envelope padding penalty, SPLERGE's primary operating point is evaluated at $\text{IoU} \ge 0.25$ (at $\text{IoU} \ge 0.50$, loose grid tiles rarely match tight word crops).
   - The core **H1 destructive repair finding is completely threshold-independent**: it measures the internal loss from Stage 1 (Raw Split) to Stage 2 (Merge). At $\text{IoU} \ge 0.25$, merge destroys **34.59% of valid cells** ($1,833 \rightarrow 1,203$). At $\text{IoU} \ge 0.50$, merge destroys **46.97% of valid cells** ($264 \rightarrow 144$). In both cases, 72% of tables suffer net TP loss, and 0% gain TPs.
4. **Reconciliation of Flagship Example (`_554187`):**
   - In `splerge_per_table_metrics.csv` (Row 4), `pubtabnet_tok_554187` reproduces the standalone smoke test metrics exactly: **GT=72**, **Raw TP@0.25=70 (97.22% recall)**, **Post TP@0.25=38 (52.78% recall)**, **$\Delta \text{TP} = -32$**.
   - `_554187` ranks #2 of 100 in the sample because its dense text fills almost the entire cell tile, naturally avoiding the envelope padding penalty.



