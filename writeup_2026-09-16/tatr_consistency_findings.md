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
| **Ordinary Overlaps per Table** | Mean = 3.6, Median = 2.0 | Mean = 0.53, Median = 0.0 | When overlaps occur in TATR, they are concentrated in dense tables (>40 rows) |
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
- Overlaps were exclusively among row predictions (`raw_col_iou_overlaps = 0` across all 100 tables). Column strips are well-separated, but horizontal row queries collide in vertically dense tables.

### B. Absence of Catastrophic Collapse (0.0% vs 5.0% in TableFormer)
- TableFormer displayed a distinct heavy-tailed failure mode where the decoder emitted up to 345 bounding boxes spanning the entire table canvas (IQR fence = 24 overlaps; max observed was 1,128 overlaps in `_629558`).
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
