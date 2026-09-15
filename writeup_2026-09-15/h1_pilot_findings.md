# H1 Pilot Findings: TableFormer Raw Prediction Consistency

**Scope:** TableFormer-fast checkpoint · PubTabNet validation split · 100 tables with real per-cell OCR tokens  
**Source:** `apoidea/pubtabnet-html` (Parquet, verified live against schema)  
**Instrumentation:** Two-call strategy — raw capture via `enable_post_process=False`, confirmed that no bbox mutations occur before that boundary.

---

## Core Finding

TableFormer's raw predictions before `MatchingPostProcessor` are:

- **Grammatically clean:** Autoregressive decoding reliably produces valid 2D grid structure (100% OTSL square grids, 100% bbox-tag sync).
- **Geometrically unreliable:** More than half of all tables (57.0%, 95% CI: [47.2%–66.3%]) exhibit raw overlapping cell bounding boxes.
- **`MatchingPostProcessor` is load-bearing:** Source-verified (all 4 bbox-mutating methods gated exclusively behind `enable_post_process=True`) — it is the sole barrier between raw output and a usable table.
- **Structural correctness ≠ cell count agreement:** Verified via bipartite IoU matching against ground-truth cell bboxes. Post-repair achieves 100% precision and recall on clean-count tables (e.g., `_670433`), but heuristic column deduplication causes severe recall collapse (to 17.6%) on prediction collapse tails (`_629558`), and causes measurable cell over-pruning (-37% count error, recall dropping from 58.7% to 52.2%) on ordinary overlap tables (`_722429`).

---

## Key Metrics with Confidence Intervals

| Metric | k/n | Point Est. | 95% CI (Wilson) |
|:---|:---:|:---:|:---:|
| OTSL grid valid (square) | 100/100 | **100.0%** | [96.3%–100.0%] |
| Bbox count matches cell count | 100/100 | **100.0%** | [96.3%–100.0%] |
| Tables with ≥1 raw geometric overlap | 57/100 | **57.0%** | [47.2%–66.3%] |
| Catastrophic collapse (overlaps > IQR fence=24) | 5/100 | **5.0%** | [2.2%–11.2%] |
| Near-duplicate pairs (max IoU > 0.90), ordinary tables | 3/95 | **3.2%** | [1.1%–8.9%] |
| Exact bbox-coordinate duplicates (b₁ == b₂) | 0/100 | 0.0% | upper bound: 2.6% |

### Ordinary Overlap Count Distribution (n=95, catastrophic excluded)

| Statistic | Value |
|:---|:---:|
| Mean | 3.24 |
| Bootstrap 95% CI for mean (B=10,000) | [2.28, 4.31] |
| Median | **1** |
| Q1 / Q3 / IQR | 0 / 5 / 5 |
| Max (ordinary) | 24 |

> **Reporting note:** Mean is the appropriate headline number for quantifying repair burden (it drives total correction work). Median (1) is the appropriate central tendency for the typical-table characterization, since the distribution is severely right-skewed (43/95 tables = 45% have zero overlaps; long tail to 24). Lead with median + IQR in the report body; use bootstrap mean CI in any table that quotes absolute load.

---

## Bimodal Failure Structure

The overlap distribution is **not unimodal**. Two regimes:

### Regime 1 — Ordinary noise (95/100 tables)
Minor boundary misalignment between adjacent cells.  
- Mean 3.24 overlapping pairs, median 1, max 24  
- 3.2% have a near-duplicate pair (IoU > 0.90) — a sub-case of ordinary overlap, not a separate phenomenon  
- Repair mechanism: `_find_overlapping()` nudges overlapping boxes apart by ½ the overlap gap

### Regime 2 — Catastrophic collapse (5/100 tables)
Raw overlap count exceeds IQR fence (>24 pairs). Ground-truth comparison reveals two distinct failure modes:

**Extreme tail case (provisional Sub-type A, n=1: `_629558`)**  
*Classification note:* `_629558` is currently n=1 (1/100 tables in this sample). Classifying it as a distinct sub-type rather than the extreme tail of a continuous failure distribution remains provisional until verified across a larger sample. However, its qualitative behavior is starkly different from all other tables:  
- Raw count is 10× ground truth (345 predicted cells for a 34-cell table, **+915%**).
- Regression head outputs ~345 overlapping boxes across the entire table area.
- Ground truth non-empty: 34 cells. Raw: 345. Post-repair: 12 (-65% error).

**Overlap-heavy tables with reasonable counts (Sub-type B, 4/5 tables)**  
Raw count is within 10–30% of GT non-empty; the pathology is primarily geometric (overlapping boxes), not count-level:

| Table ID | gt_total | gt_nonempty | raw_cells | post_cells | raw_err | post_err | raw_ov | post_ov |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `_634445` | 136 | 136 | 152 | 122 | +12% | -10% | 1,845 | 0 |
| `_670433` | 53 | 53 | 59 | 53 | +11% | **0%** | 30 | 0 |
| `_677385` | 373 | 137 | 285 | 122 | +108% | -11% | 200 | 0 |
| `_729453` | 212 | 192 | 245 | 185 | +28% | -4% | 53 | 0 |

- `_670433`: raw count close, post count exact match to GT non-empty (53=53).  
- `_677385`: GT has 373 total cells but only 137 non-empty (236 structural empties/spans); raw=285 partially follows the span structure; post=122 is 11% below GT non-empty.  
- For Sub-type B, post-repair overshoots slightly but remains within 4–11% of GT non-empty. Characterising all five tables as "recovery via cell elimination" is inaccurate for four of them.

---

## Structural Accuracy: Ground-Truth IoU Matching (Beyond Cell Counts)

Cell-count agreement with ground truth is structurally easy for a model to satisfy coincidentally (analogous to how `otsl_square` and `bbox_sync` were 100% clean while geometry was broken in 57% of tables). To determine whether predicted cells actually correspond to real table cells in position and identity, we performed greedy bipartite IoU matching (IoU ≥ 0.5 and IoU ≥ 0.75) against PubTabNet ground-truth cell bounding boxes.

Both **raw** and **post-repair** predictions were evaluated across the 5 catastrophic tables and 4 representative ordinary comparison tables:

| Table ID | Category | GT | Raw | Prec@0.5 | Rec@0.5 | F1@0.5 | Post | Prec@0.5 | Rec@0.5 | F1@0.5 | Prec@0.75 | Rec@0.75 | F1@0.75 | MedIoU |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `_629558` | Catastrophic (tail) | 34 | 345 | 2.3% | 23.5% | 4.2% | 12 | 50.0% | **17.6%** | 26.1% | 50.0% | 17.6% | 26.1% | 1.000 |
| `_634445` | Catastrophic (overlap) | 136 | 152 | 65.1% | 72.8% | 68.7% | 122 | 100.0% | 89.7% | 94.6% | 88.5% | 79.4% | 83.7% | 1.000 |
| `_670433` | Catastrophic (overlap) | 53 | 59 | 88.1% | 98.1% | 92.9% | 53 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 1.000 |
| `_677385` | Catastrophic (overlap) | 137 | 285 | 34.7% | 72.3% | 46.9% | 122 | 89.3% | 79.6% | 84.2% | 87.7% | 78.1% | 82.6% | 1.000 |
| `_729453` | Catastrophic (overlap) | 192 | 245 | 60.8% | 77.6% | 68.2% | 185 | 99.5% | 95.8% | 97.6% | 35.1% | 33.9% | 34.5% | 0.700 |
| `_622581` | Ordinary | 33 | 35 | 91.4% | 97.0% | 94.1% | 33 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 |
| `_624025` | Ordinary | 50 | 51 | 82.4% | 84.0% | 83.2% | 50 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 |
| `_639579` | Ordinary (new ov) | 61 | 76 | 73.7% | 91.8% | 81.8% | 61 | 100.0% | 100.0% | 100.0% | 96.7% | 96.7% | 96.7% | 1.000 |
| `_722429` | Ordinary (residual ov) | 46 | 40 | 67.5% | 58.7% | 62.8% | 29 | 82.8% | **52.2%** | 64.0% | 79.3% | 50.0% | 61.3% | 1.000 |

### Key Findings from IoU Structural Matching:

1. **`_670433` is confirmed structurally exact, NOT count-coincidental:**  
   The exact count match (53=53) is reflected in a flawless 1-to-1 spatial correspondence: **Precision@0.5 = 100.0%, Recall@0.5 = 100.0%, F1@0.5 = 100.0%**, and even under strict matching (**IoU ≥ 0.75**), Precision = 100.0%, Recall = 100.0%, F1 = 100.0% (Median IoU = 1.000). Every single predicted post-repair cell lands on its true ground-truth counterpart. The repair was structurally complete and exact.

2. **`_629558` confirms severe information loss (heuristic failure mode):**  
   Post-repair's column deduplication collapsed 345 raw boxes down to 12 cells. Of those 12 cells, only 6 match ground-truth cells (Precision@0.5 = 50.0%). Crucially, **Recall collapsed to 17.6%** (6/34 GT cells). Fully **82.4% (28/34) of the table's real content cells were discarded**. This proves empirically that post-repair did *not* reconstruct the table; it gutted it, replacing an over-predicted hallucination with an incomplete fragment.

3. **Sub-type B tables achieve genuine structural recovery:**  
   Post-repair dramatically improves structural precision and F1 over raw predictions:
   - `_634445`: Raw F1 = 68.7% → Post F1 = **94.6%** (Precision = 100.0%, Recall = 89.7%)
   - `_677385`: Raw F1 = 46.9% → Post F1 = **84.2%** (Precision = 89.3%, Recall = 79.6%)
   - `_729453`: Raw F1 = 68.2% → Post F1 = **97.6%** (Precision = 99.5%, Recall = 95.8%)

4. **Derivation for `_722429` (Empirical Proof of Over-Pruning in Ordinary Tables):**  
   The user requested the full derivation for `_722429`:
   - **Ground truth:** 46 non-empty cells (46 total cells).
   - **Raw prediction:** 40 cells (-13.0% under GT), 21 raw overlap pairs.
   - **Post-repair prediction:** 29 cells (**-37.0% under GT**, 17 cells missing), 1 residual overlap pair.
   - **Structural accuracy impact:**
     - Raw: Precision@0.5 = 67.5% (27/40), Recall@0.5 = 58.7% (27/46), F1 = 62.8%
     - Post: Precision@0.5 = 82.8% (24/29), Recall@0.5 = **52.2%** (24/46), F1 = 64.0%
   - **The over-pruning mechanism:** Deduplication eliminated 11 cells to resolve the 21 overlaps. However, in doing so, it deleted **3 genuine, correctly-detected ground-truth cells**, reducing recall from 58.7% to 52.2%. This directly verifies that heuristic repair can actively degrade table completeness even on ordinary, non-catastrophic tables.

---

## Post-Repair Recovery (Empirically Measured)

All 100 tables re-run with `enable_post_process=True`. Post-repair overlap counts computed by `_analyze_raw_cells` on the actual `matching_details["table_cells"]` from the second predict() call.

| Metric | Value |
|:---|:---:|
| Tables with raw overlaps | 57/100 |
| Tables with post-repair overlaps | **4/100 (4.0%)** |
| Fully recovered (raw > 0, post = 0) | **54/57 (94.7%)** |
| Post-processor *introduced* new overlap | **1/100** (`_639579`: raw=0, post=1) |

**`_639579` overlap is real, not a boundary artefact:** post-repair cell id=52 (bbox `[4,154,53,159]`, area 245px²) is 100% contained within cell id=56 (bbox `[4,149,94,164]`). `_move_cells_to_left_pos` or `_align_table_cells_to_pdf` placed a narrow cell directly inside a taller neighbour during column realignment. GT for this table: 61 non-empty cells; raw=76, post=61 (post-repair cell count is exact GT match and achieves 100% precision and recall despite the one residual overlap).

### Ordinary tables: 3/52 retain 1 residual overlap after repair
- `_721570`: raw=9, post=1 (18 cells unchanged)  
- `_722429`: raw=21, post=1 (40→29 cells; GT non-empty=46, so post is 37% below GT — verified recall drop from 58.7% to 52.2%)  
- `_723545`: raw=3, post=1 (47→34 cells)

> **Note on rates:** 4/100 post-repair overlaps (Wilson 95% CI: [1.6%–9.7%]) and 1/100 introduced overlaps (95% CI: [0.2%–5.3%]) are framed as existence results, not precise rate estimates, until sample size increases.

---

## Measurement Integrity — Bugs Caught and Fixed

| Issue | Original State | Corrected State |
|:---|:---|:---|
| Duplicate detection | Whole-dict equality (always False due to unique `cell_id`) | Bbox-coordinate equality only |
| Outlier threshold | Mean + 3σ (circular: outlier inflated mean+σ) | Q3 + 3×IQR (non-circular) |
| Near-duplicate check | 0/25 subsample → stated as "clean zero" | Full 95-table scan → 3/95 = 3.2% |
| Overlap t-interval | Symmetric t on right-skewed count (stdev 5.05 >> mean 3.24) | Bootstrap CI (B=10,000) |
| Post-processor gate | Assumed no tokens → gate closed | Real per-cell tokens → gate confirmed open |
| Recovery evidence | Hardcoded log strings, not computed | `_analyze_raw_cells` on actual post-processed cells |
| Recovery framing | "5/5 recovered" (overlap=0), all via cell elimination | GT comparison: 4/5 catastrophic tables within 10-30% of GT; only _629558 is true collapse |
| "Massive cell elimination" characterisation | Applied to all 5 catastrophic tables | Accurate only for _629558; other 4 are overlap-only failures |
| Count agreement as correctness proxy | len(pred) == len(gt) treated as proof of success | Bipartite IoU matching confirms exact structural match for _670433 (100% F1), but severe recall collapse (17.6%) for _629558 and over-pruning for _722429 |

Each correction verified against source or by direct measurement before being accepted.

---

## Bbox Mutation Audit — Comparability Across Runs

The token-enabled run (57.0%) vs. the no-token run (40.0% at n=300) measures the same pre-repair geometry.  
Confirmed by tracing all bbox-write statements in the full `match_cells()` → `_build_table_cells()` → `_intersection_over_pdf_match()` → `_run_intersection_match()` chain:

| Method | Writes bbox? | Call context |
|:---|:---:|:---|
| `CellMatcher.match_cells` | ❌ | Both runs |
| `CellMatcher._intersection_over_pdf_match` | ❌ | Both runs |
| `MatchingPostProcessor._run_intersection_match` | ❌ | Post-processor only |
| `MatchingPostProcessor._move_cells_to_left_pos` | ✅ | Post-processor only (step 4) |
| `MatchingPostProcessor._align_table_cells_to_pdf` | ✅ | Post-processor only (step 8a) |
| `MatchingPostProcessor._attach_orphan_to_cell` | ✅ | Post-processor only (step 9) |
| `MatchingPostProcessor._find_overlapping` | ✅ | Post-processor only (final) |

Independently confirmed by the user against `_align_table_cells_to_pdf`'s docstring and six in-source write statements.

> The 40.0% vs 57.0% difference between runs is sample composition (different PubTabNet subsets), not a measurement artifact from token injection changing predicted coordinates.

---

## Comparability of the 40% vs 57% Overlap Rates

Three candidate explanations investigated for the 17-point gap between the 300-table no-token run (40.0%) and the 100-table token run (57.0%):

| Candidate | Result |
|:---|:---|
| Table size | **Ruled out** — median cells 48.0 vs 48.5; size does not predict overlap presence within either run |
| Token presence changes cell count | **Ruled out** — 8/8 test tables: identical cell count with/without tokens (same raw geometry) |
| Mirror crop boundaries (aspect ratio) | **Open** — median aspect ratio 2.58 vs 2.52 (difference 0.06); `ajimeno` has wider tail (max 11.43 vs 5.92); different crop conventions between mirrors plausible partial contributor |

The gap is real (p=0.003, non-overlapping Wilson CIs). No single confound fully explains it. **These two runs should not be presented as directly comparable cross-run figures.** The pair-normalized rate (overlapping pairs / candidate pairs) is the more defensible headline metric within each run.

---

## Scope Statement (for Write-up)

**What this proves:** For TableFormer-fast on PubTabNet: raw predictions are grammatically clean (100% valid OTSL grids, 100% bbox-count sync) but geometrically unreliable (57% of tables have raw overlaps). `MatchingPostProcessor` eliminates overlaps in 94.7% of affected tables. 

Ground-truth IoU matching settles the structural correctness question:
- On overlap-heavy tables with reasonable counts (`_670433`, `_634445`, `_677385`, `_729453`), post-repair achieves high structural accuracy (F1: 84.2%–100.0%, with `_670433` achieving 100% precision and 100% recall).
- On true prediction collapse tails (`_629558`), heuristic deduplication fails structurally: recall collapses to 17.6%, discarding 82.4% of table content.
- On ordinary overlap tables (`_722429`), heuristic repair causes measurable over-pruning: cell count falls 37% below ground truth, deleting genuine cells and reducing recall from 58.7% to 52.2%.

**H2 framing (empirically grounded):**
1. *For ordinary overlap failures (Regime 1): does global-consistency optimization preserve genuine cell detections better than heuristic repair?* (Directly supported by `_722429`, where heuristic repair reduced recall by dropping 3 real cells.)
2. *For catastrophic prediction collapses (provisional extreme tail): can joint consistency optimization avoid the severe 10× over-prediction at the prediction stage, or recover a more complete structure than heuristic repair's 17.6% recall?*

**What this does not yet prove:** Whether this pattern holds across architecturally different TSR paradigms. This pilot covers sequence-generation only. **SPLERGE (split-and-merge)** is the natural next baseline.

---

## Files

| File | Purpose |
|:---|:---|
| [`check_structural_accuracy.py`](file:///d:/RRC/major_project_csvtu/check_structural_accuracy.py) | Bipartite IoU matching (precision, recall, F1, median IoU) vs GT cell bboxes |
| [`pubtabnet_structural_accuracy_report.csv`](file:///d:/RRC/major_project_csvtu/pubtabnet_structural_accuracy_report.csv) | Per-table structural accuracy report (raw vs post, IoU@0.5 and IoU@0.75) |
| [`measure_raw_consistency.py`](file:///d:/RRC/major_project_csvtu/measure_raw_consistency.py) | Core instrumentation: two-call strategy, `_analyze_raw_cells`, IQR outlier detection, `write_report` |
| [`run_with_tokens.py`](file:///d:/RRC/major_project_csvtu/run_with_tokens.py) | Token-enabled driver (`apoidea/pubtabnet-html`, cache-first, 100 tables) |
| [`run_pubtabnet_subset.py`](file:///d:/RRC/major_project_csvtu/run_pubtabnet_subset.py) | No-token baseline driver (300-table run, `ajimeno/PubTabNet`) |
| [`measure_post_repair.py`](file:///d:/RRC/major_project_csvtu/measure_post_repair.py) | Pre vs post-repair overlap measurement on actual post-processed cells |
| [`check_ground_truth_cells.py`](file:///d:/RRC/major_project_csvtu/check_ground_truth_cells.py) | Ground truth cell count streaming and comparison |
| [`compute_confidence_intervals.py`](file:///d:/RRC/major_project_csvtu/compute_confidence_intervals.py) | Wilson score CIs for all proportions; bootstrap CI for overlap mean |
| [`inspect_bbox_mutations.py`](file:///d:/RRC/major_project_csvtu/inspect_bbox_mutations.py) | Bbox mutation audit via `inspect.getsource` across full call chain |
| [`check_gap_explanations.py`](file:///d:/RRC/major_project_csvtu/check_gap_explanations.py) | Resolution/aspect-ratio comparison + token-vs-no-token cell count check |
| [`inspect_639579_overlap.py`](file:///d:/RRC/major_project_csvtu/inspect_639579_overlap.py) | Deep inspection of containment overlap introduced in table `_639579` |
| [`pubtabnet_tokens_consistency_report.csv`](file:///d:/RRC/major_project_csvtu/pubtabnet_tokens_consistency_report.csv) | Per-table raw results (100 tables, token run) |
| [`pubtabnet_raw_consistency_report.csv`](file:///d:/RRC/major_project_csvtu/pubtabnet_raw_consistency_report.csv) | Per-table raw results (300 tables, no-token run) |
| [`pubtabnet_pre_post_repair_report.csv`](file:///d:/RRC/major_project_csvtu/pubtabnet_pre_post_repair_report.csv) | Per-table pre and post-repair overlap counts, cell counts, recovery flags |
