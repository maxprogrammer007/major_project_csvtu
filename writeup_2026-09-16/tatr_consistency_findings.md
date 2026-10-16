# Paradigm 2 Evaluation: Microsoft Table Transformer (TATR-v1.1-All) on PubTables-1M

**Date:** 2026-09-16  
**Model:** `microsoft/table-transformer-structure-recognition-v1.1-all` (ResNet-18 + DETR structure recognition)  
**Dataset:** PubTables-1M (Official Microsoft test set, native in-domain evaluation)  
**Sample Size:** $N = 100$ tables (100 images, 1,261 GT rows, 576 GT columns, 202 GT spanning cells)  
**Report File:** [`tatr_pubtables1m_consistency_report.csv`](file:///d:/RRC/major_project_csvtu/tatr_pubtables1m_consistency_report.csv)  
**Execution Script:** [`measure_tatr_consistency.py`](file:///d:/RRC/major_project_csvtu/measure_tatr_consistency.py)

---

## 1. Executive Summary & Cross-Paradigm Comparison

The central question driving this multi-paradigm study is whether the structural failure pattern identified in TableFormer (Paradigm 1: Sequence Generation)—namely, **grammatically clean syntax masking severe geometric inconsistency, paired with destructive heuristic repair**—is unique to autoregressive sequence decoding or systematic across TSR paradigms.

With Paradigm 2 (Object Detection / DETR) instrumented on its native domain (PubTables-1M), the results provide a crisp, evidence-backed architectural comparison:

| Metric / Dimension | Paradigm 1: Sequence Generation (TableFormer-fast) | Paradigm 2: Transformer / DETR (TATR-v1.1-All) | Statistical Contrast & Interpretation |
|:---|:---:|:---:|:---|
| **Underlying Architecture** | Autoregressive sequence decoding (OTSL tags + 2D cell coordinate regression) | Set prediction via learned object queries (DETR with Hungarian bipartite loss) | Sequence generation predicts individual cells; DETR predicts macro row/col strips |
| **Benchmark Dataset** | PubTabNet (`apoidea/pubtabnet-html`) | PubTables-1M (Test split, native domain) | *Methodological note: Datasets are domain-specific to avoid conflating architecture with distribution shift.* |
| **Raw Overlap Incidence Rate** | **57.0%** [95% CI: 47.2% – 66.3%] | **10.0%** [95% CI: 5.5% – 17.4%] | Statistically significant reduction ($z = 7.15, p < 10^{-12}$); DETR's set prediction enforces macro separation |
| **Catastrophic Collapse Rate** | **5.0%** [95% CI: 2.2% – 11.2%] | **0.0%** [95% CI: 0.0% – 3.7%] | Autoregressive looping produces degenerate collapse; DETR's fixed query budget (125) prevents runaway repetition |
| **Ordinary Overlaps per Table** | Mean = 3.24 [bootstrap 95% CI 2.28–4.31], Median = 1 | Mean = 0.53, Median = 0.0 | When overlaps occur in TATR, they are concentrated in dense tables (>40 rows) |
| **Heuristic Repair Mechanism** | `MatchingPostProcessor` (`_deduplicate_cells` + `_align_table_cells_to_pdf`) | `tatr_postprocess.py` (`refine_rows/columns` via NMS + `align_rows/columns`) | Both paradigms rely on post-hoc heuristic repair rather than intrinsic architectural consistency |
| **Heuristic Repair Side Effect** | **Destructive over-pruning**: drops true cells (e.g. `_722429` recall 58.7% $\to$ 52.2%) | **Destructive over-pruning**: drops true rows (e.g. `PMC6127715` recall 100% $\to$ 88.1%) | **Systematic across paradigms:** Heuristic NMS discards genuine structures in dense/adjacent layouts |
| **Circularity in GT Comparison** | **Severe**: Post-repair `MedIoU=1.000` is an artifact of snapping to GT-derived OCR tokens | **None**: No token snapping; row/col extent aligned only to table boundary; MedIoU reflects true model geometry | TATR post-repair MedIoU = 0.947 (rows) and 0.970 (columns) are genuine geometric metrics |

---

## 2. Source-Traced Repair Boundary Audit (TATR)

Following the rigorous verification standard established in Phase 1, we traced the exact code boundary separating raw model predictions from heuristic mutations in `microsoft/table-transformer`:

```
Raw Model Output
  │ (ResNet-18 Backbone + Transformer Encoder-Decoder)
  ▼
inference.py: outputs_to_objects()
  │  - Argmax over pred_logits
  │  - Box unnormalization & scaling to image coordinates
  │  [RAW BOUNDARY: Captured here before any thresholding/NMS/alignment]
  ▼
postprocess.py: objects_to_table_structures()
  ├─ refine_rows() / refine_columns()
  │    └─ NMS (match_criteria="object2_overlap", threshold=0.5 for rows, 0.25 for columns)
  │    └─ [HEURISTIC PRUNING STEP: Over-prunes genuine rows in dense tables]
  ├─ align_rows() / align_columns()
  │    └─ Mutates bboxes in place: stretches row left/right and col top/bottom to table bounds
  │    └─ [BBOX MUTATION STEP: Enforces physical strip coverage, not token snapping]
  └─ refine_table_structures()
       └─ Supercell alignment & NMS
  ▼
postprocess.py: table_structure_to_cells()
  └─ Geometric Cartesian intersection: cell_rect = row_rect.intersect(column_rect)
```

### Key Verification: Why TATR is Immune to Token Circularity
In TableFormer, `MatchingPostProcessor._align_table_cells_to_pdf` takes token bounding boxes (derived from ground truth in our PubTabNet test pipeline) and directly overwrites the predicted cell coordinates.

In TATR:
- When evaluated with `tokens=[]` (the visual TSR mode), `refine_rows` uses pure geometric overlap NMS (`match_criteria="object2_overlap"`).
- `align_rows` and `align_columns` do **not** consult tokens at all; they merely stretch row widths to the table bounding box.
- Therefore, TATR's post-repair coordinates are completely independent of token ground truth. The resulting row MedIoU of 0.947 and column MedIoU of 0.970 represent true spatial regression accuracy, not an evaluation artifact.

---

## 3. Detailed Findings on Paradigm 2 (TATR)

### A. Raw Structural Overlaps (10.0% Incidence Rate)
Out of 100 tables evaluated:
- **90 tables (90.0%)** were completely clean of intra-class row or column overlaps.
- **10 tables (10.0%, 95% CI: [5.5%, 17.4%])** exhibited geometric overlaps among raw predictions.
- Overlaps occurred in both rows and columns: 7 tables exhibited row overlaps and 4 tables exhibited column overlaps (1 table had both). While row overlaps were more frequent in dense tables, column overlaps do exist (5 column pairs overlapped across the 4 tables), disproving the assumption that DETR's column strips never collide.

### B. Absence of Catastrophic Collapse (0.0% vs 5.0% in TableFormer)
- TableFormer displayed a distinct heavy-tailed failure mode where the decoder emitted up to 345 bounding boxes spanning the entire table canvas (IQR fence = 24 overlaps; max observed was 33,766 overlaps in `_629558`).
- In TATR, the maximum number of overlapping pairs observed in any single table was 20 (in `PMC4962420_table_0`, a table with 47 dense rows).
- Applying the identical IQR fence formula ($Q_3 + 3 \times \text{IQR}$), the fence is $\infty$ (since $Q_1 = 0, Q_3 = 0, \text{IQR} = 0$). Even using an absolute threshold of $> 20$ overlaps, **0/100 tables (0.0%, 95% CI: [0.0%, 3.7%])** exhibited catastrophic runaway collapse.
- **Architectural Reason:** DETR decodes a fixed number of learned object queries (125 queries). The bipartite matching loss during training explicitly suppresses non-matched queries toward the "no-object" background class. The model literally cannot produce unbounded recursive box duplication.

### C. The Destructive Nature of Heuristic Repair (Empirical Proof in TATR)
While TATR avoids catastrophic runaway collapse, its heuristic repair pipeline exhibits the exact same flaw as TableFormer's `_deduplicate_cells`: **destructive over-pruning**.

Across the 100 tables, heuristic NMS discarded **7 valid True Positive rows** that had correctly matched ground truth annotations at IoU $\ge 0.5$:

#### Case Study: `PMC6127715_table_0` (Dense 42-Row Table)
- **Ground Truth:** 42 rows, 7 columns.
- **Raw Predictions:** 50 rows. 42 of them matched Ground Truth exactly (True Positives = 42/42, **Recall = 100.0%**). The remaining 8 were low-confidence boundary duplicates.
- **Raw Inconsistency:** 19 raw overlapping row pairs.
- **Heuristic Repair Effect:** `refine_rows()` applied NMS. Instead of merely pruning the 8 duplicates, it suppressed **5 genuine rows**:
  - Repaired True Positives fell to 37.
  - Row Recall dropped from **100.0% to 88.1%** (a 11.9% loss of ground-truth content).
  - Row F1 dropped from **91.3% to 85.1%**.
- **Incomplete Repair:** Despite destroying 5 genuine rows, the repair pipeline *still* left **9 overlapping row pairs** in the final table!

#### Case Study: `PMC4962420_table_0` (47-Row Table)
- **Ground Truth:** 47 rows, 2 columns.
- **Raw Predictions:** 54 rows (42 True Positives, 20 overlapping pairs).
- **Heuristic Repair Effect:** Pruned 1 True Positive row (TP dropped from 42 to 41); left **14 overlapping pairs** in the final post-repair table.

**Conclusion on Heuristic Repair:** NMS is a blunt computer vision heuristic designed for isolated object instances (e.g. cars, pedestrians), not tightly stacked, contiguous 1D structural grids. In dense tables, adjacent rows naturally have small IoU/boundary contact that NMS cannot distinguish from duplicate detections without semantic content.

---

## 4. Ground Truth Structural Accuracy (Bipartite IoU Matching)

Using bipartite greedy IoU matching at threshold $\tau = 0.5$:

| Structural Element | Metric | Raw Prediction | Post-Repair Prediction | Impact of Repair |
|:---|:---|:---:|:---:|:---:|
| **Rows** | **Micro Precision** | 97.3% | 97.6% | +0.3% |
| | **Micro Recall** | **99.2%** | **98.7%** | **-0.5% (Over-pruning)** |
| | **Micro F1** | 98.23% | 98.15% | -0.08% |
| | **Macro F1** | 99.16% | 99.13% | -0.03% |
| | **Median Matched IoU** | 0.9484 | 0.9473 | -0.0011 |
| **Columns** | **Micro Precision** | 98.8% | 99.5% | +0.7% |
| | **Micro Recall** | 99.7% | 99.7% | 0.0% |
| | **Micro F1** | 99.22% | 99.57% | +0.35% |
| | **Macro F1** | 99.15% | 99.45% | +0.30% |
| | **Median Matched IoU** | 0.9686 | 0.9699 | +0.0013 |

---

## 5. Summary of Conclusions for the Research Report

1. **Paradigm Contrast Confirmed:** The high raw inconsistency (57%) and catastrophic collapse mode (5%) seen in TableFormer are largely artifacts of unconstrained autoregressive sequence generation. The DETR object detection paradigm reduces raw inconsistency to 10% and eliminates catastrophic collapse entirely.
2. **The "Destructive Heuristic Repair" Hypothesis Generalizes:** In both paradigms, post-processing heuristics (whether OTSL cell deduplication or DETR row NMS) systematically destroy valid table structure in dense tables. Repair is an unprincipled band-aid that sacrifices recall for cosmetic cleanliness.
3. **Measurement Integrity Maintained:** By isolating the raw-to-repair boundary in `tatr_postprocess.py` and running with `tokens=[]`, we proved that TATR's high geometric precision (MedIoU $\approx 0.95$) is intrinsic to the DETR regression head and completely free of the token-snapping circularity that compromised TableFormer's post-repair evaluation.
4. **Checkpoint Quality Verification:** The extremely high structural extraction rate on the raw grid (99.2% row recall, 99.7% column recall) is reassuringly consistent with the Table Transformer paper's published performance on PubTables-1M. While our bipartite matching metric is not a strict one-to-one equivalent to the paper's GriTS topology score (0.96–0.98), the results demonstrate this is a healthy, state-of-the-art model without the severe degradation confounds observed in Paradigm 4.

---

## 6. H2 Extension: Exact MWIS Optimization vs. Greedy NMS

In Paradigm 4 (SPLERGE), replacing a greedy heuristic with exact Set Packing recovered 99.6% of destroyed True Positives. To test if this generalized to Paradigm 2, we formulated TATR's row/column selection as a **Maximum Weight Independent Set (MWIS)** problem via Integer Linear Programming (HiGHS). 

- **Formulation:** A conflict graph was built using TATR's exact NMS criteria (`object2_overlap > 0.5` for rows). The objective maximized the sum of DETR confidence scores for the selected non-overlapping boxes.
- **Hypothesis:** MWIS would rescue genuine rows by favoring the combined score of multiple small genuine rows (e.g., $0.90 + 0.85 = 1.75$) over a single large false-positive duplicate (e.g., $0.95$) that would otherwise suppress them in greedy NMS.

### Results: The Negative Finding with Confidence Scores
On the 100-table test set, exact MWIS yielded **negligible (+1 TP) structural lift** over greedy NMS when optimizing the model's own confidence scores:
- **Heuristic NMS TP:** 1,244
- **MWIS ILP TP:** 1,245

In `PMC6127715_table_0`, where NMS destroyed 5 genuine rows, MWIS *also* destroyed 5 genuine rows (swapping exactly 1 TP for another). 

To verify *why* greedy and optimal coincided so perfectly, we analyzed the conflict graph across all 100 tables. Out of **1,272 connected components** in the conflict graph, only **2 components (0.16%)** contained more than 2 nodes. 

This confirms the structural difference between the paradigms: because DETR predicts macroscopic rows directly, conflicts are almost entirely isolated 1-to-1 overlaps between a genuine row and a slightly shifted false duplicate. In a graph composed of disjoint edges, greedy NMS is mathematically guaranteed to find the exact Maximum Weight Independent Set. 

### Experiment 2: The Limits of a Naive Geometric Objective
The negative result above proves that optimization cannot help *if the objective is just the model's own confidence*. In SPLERGE, exact optimization only succeeded because it used a content-aware signal (token containment count) rather than the model's naive weights.

To test if optimization could rescue TATR rows when decoupled from confidence scores, we ran a second MWIS experiment across the full 100-table dataset. As a proxy for content (lacking direct OCR tokens), we replaced the DETR confidence scores with a **structural regularity objective**: the weight of each row was set inversely proportional to its height's distance from the table's median row height. 

**Full Corpus Results:**
- **Heuristic NMS TP:** 1,244 (-7 destroyed)
- **MWIS (Height Objective) TP:** 1,245 (-6 destroyed)

While the net lift remains +1 TP, the underlying mechanics reveal exactly what the user anticipated: an explicit win/loss trade-off.
- **The Wins:** In the flagship worst-offender table `PMC6127715_table_0`, the geometric objective successfully recovered **3 out of the 5** genuine rows that confidence-based NMS destroyed. 
- **The Losses (Cost):** The new objective actively introduced **2 new destructions** in tables that NMS had handled perfectly (`PMC5762819_table_1` and `PMC5897469_table_0`), punishing genuine rows simply for having a legitimate natural variance from the median height.

**The Headline Mechanism: The Geometric Proxy's Microscopic Brittleness**
Why did the regularity objective fail to recover the remaining 2 rows in `PMC6127715`, and why did it introduce new destructions in `PMC5762819` and `PMC5897469`? By inspecting the solver's decisions, we found that all four failures share the exact same mechanistic flaw: the false duplicates produced by DETR were slightly tighter crops that mathematically sat fractions of a pixel closer to the global median height than the ground-truth rows themselves.

For example, in `PMC5897469_table_0`, the median row height was 16.01:
- Genuine Row height: 16.06 (DETR Confidence: 0.993)
- Duplicate height: 16.03 (DETR Confidence: 0.653)

Under greedy NMS, the model got this perfectly right, selecting the genuine row due to its 99% confidence. But because the geometric objective blindly minimized distance to the median, it assigned the duplicate a strictly higher score (0.999 vs 0.996). The exact solver was forced to discard the genuine row in favor of a low-confidence duplicate. The geometric proxy is so brittle that microscopic proximity to the median overrides a 30-point margin in true model confidence. 

*(Note: In all experiments, the definition of a "conflict" remained strictly preserved. The ILP used TATR's exact `object2_overlap > 0.5` criteria; only the weight function was swapped.)*

### Conclusion
In Paradigm 2, the TP destruction is **not caused by an algorithmic flaw in the greedy heuristic**. Because DETR predicts macroscopic rows directly, conflicts are overwhelmingly 1-to-1 pairwise overlaps (verified at 99.8% across the corpus). In this topology, greedy NMS natively finds the exact Maximum Weight Independent Set. 

When TATR drops a genuine row, it does so purely because of objective miscalibration. When optimizing the model's own confidence, greedy and exact methods fail identically. When optimizing a naive geometric proxy (row height), the exact solver trades confidence errors for a brittle microscopic variance intolerance.

The ultimate lesson of H2 demonstrates, for TATR specifically, that **objective choice determines whether optimization helps or merely relocates the same kind of error**. Just as in SPLERGE, rescuing destructive repairs requires a genuine, external, content-aware signal (like token containment), not just better algorithmic machinery on top of flawed proxies.
