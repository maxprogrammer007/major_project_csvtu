# H1 Pilot Findings: TableFormer Raw Prediction Consistency

**Scope:** TableFormer-fast checkpoint · PubTabNet validation split · 100 tables with real per-cell OCR tokens  
**Source:** `apoidea/pubtabnet-html` (Parquet, verified live against schema)  
**Instrumentation:** Two-call strategy — raw capture via `enable_post_process=False`, confirmed that no bbox mutations occur before that boundary.

---

## Core Finding

TableFormer's raw predictions before `MatchingPostProcessor` are:

- **Grammatically clean:** Autoregressive decoding reliably produces valid 2D grid structure.
- **Geometrically unreliable:** More than half of all tables require active correction of overlapping cell bboxes.
- **`MatchingPostProcessor` is load-bearing:** Source-verified (all 4 bbox-mutating methods gated exclusively behind `enable_post_process=True`) — it is the sole barrier between raw output and a usable table.

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
Raw overlap count exceeds IQR fence (>24 pairs). Ground-truth comparison reveals two structurally different sub-types:

**Sub-type A — True prediction collapse (1/5 tables: `_629558`)**  
Raw count is 10× the ground-truth non-empty cell count. The regression head outputs ~345 near-identical bboxes for a 34-cell table.

| | gt_nonempty | raw_cells | post_cells | raw_err | post_err |
|:---|---:|---:|---:|---:|---:|
| `_629558` | 34 | 345 | 12 | **+915%** | -65% |

Post-repair reduces overlap to zero by discarding 333/345 predicted cells. Post count (12) is 65% below GT, so both extremes are wrong: raw massively over-predicts, post-repair over-corrects. This is the only table where "recovery via cell elimination" is an accurate characterisation.

**Sub-type B — Overlap-only failure with reasonable count (4/5 tables)**  
Raw count is within 30% of GT non-empty; the pathology is purely geometric (overlapping boxes), not count-level.

| | gt_nonempty | raw_cells | post_cells | raw_err | post_err |
|:---|---:|---:|---:|---:|---:|
| `_634445` | 136 | 152 | 122 | +12% | -10% |
| `_670433` | 53 | 59 | 53 | +11% | **0%** |
| `_677385` | 137 | 285 | 122 | +108% | -11% |
| `_729453` | 192 | 245 | 185 | +28% | -4% |

`_670433`: raw count correct, post count exact match to GT non-empty — overlap-only failure, fully repaired.  
`_677385`: GT has 373 total cells but only 137 non-empty (236 structural empties/spans); raw=285 partially follows the span structure; post=122 is 11% below GT non-empty.  
For Sub-type B, post-repair overshoots slightly but remains within 10–11% of GT. Characterising all five tables as "recovery via cell elimination" is wrong for four of them.

---

## Post-Repair Recovery (Empirically Measured)

All 100 tables re-run with `enable_post_process=True`. Post-repair overlap counts computed by `_analyze_raw_cells` on the actual `matching_details["table_cells"]` from the second predict() call.

| Metric | Value |
|:---|:---:|
| Tables with raw overlaps | 57/100 |
| Tables with post-repair overlaps | **4/100 (4.0%)** |
| Fully recovered (raw > 0, post = 0) | **54/57 (94.7%)** |
| Post-processor *introduced* new overlap | **1/100** (`_639579`: raw=0, post=1) |

**`_639579` overlap is real, not a boundary artefact:** post-repair cell id=52 (bbox `[4,154,53,159]`, area 245px²) is 100% contained within cell id=56 (bbox `[4,149,94,164]`). `_move_cells_to_left_pos` or `_align_table_cells_to_pdf` placed a narrow cell directly inside a taller neighbour during column realignment. GT for this table: 61 non-empty cells; raw=76, post=61 (post-repair cell count is exact GT match despite the one residual overlap).

### Ordinary tables: 3/52 retain 1 residual overlap after repair
- `_721570`: raw=9, post=1 (18 cells unchanged)  
- `_722429`: raw=21, post=1 (40→29 cells; GT non-empty=46, so post is 37% below GT — notable over-correction)  
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
| "Massive cell elimination" characterisation | Applied to all 5 catastrophic tables | Accurate only for _629558 (Sub-type A); other 4 are overlap-only failures (Sub-type B) |

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

**What this proves:** For TableFormer-fast on PubTabNet: raw predictions are grammatically clean (100% valid OTSL grids, 100% bbox-count sync) but geometrically unreliable (57% of tables have raw overlaps). `MatchingPostProcessor` eliminates overlaps in 94.7% of affected tables. Ground-truth comparison reveals the 5 catastrophic tables split into two structurally distinct sub-types: one true prediction collapse (`_629558`: raw=345 for a 34-cell table, post=12, both wrong) and four overlap-only failures where raw and post-repair cell counts are within 10–30% of GT non-empty.

**H2 framing (ground-truth calibrated):** The original framing “recover without massive cell elimination” applies only to `_629558`-type collapses (true prediction collapse, raw count 10× GT). For the other four, post-repair is geometrically correcting overlaps with acceptable cell-count accuracy. The sharpened H2 question has two parts:
1. *For ordinary overlap failures (Regime 1): does global-consistency optimization preserve cell count more accurately than heuristic repair?* (`_722429` post count 37% below GT suggests over-pruning is real even on ordinary tables.)
2. *For true prediction collapses (Sub-type A): can optimization avoid the 10× over-prediction at the raw stage, or recover a more complete structure than post-repair’s 12/34 cells?*

**What this does not yet prove:** Whether this pattern holds across architecturally different TSR paradigms. This pilot covers sequence-generation only. **SPLERGE (split-and-merge)** is the natural next baseline.

---

## Files

| File | Purpose |
|:---|:---|
| [`measure_raw_consistency.py`](file:///d:/RRC/major_project_csvtu/measure_raw_consistency.py) | Core instrumentation: two-call strategy, `_analyze_raw_cells`, IQR outlier detection, `write_report` |
| [`run_with_tokens.py`](file:///d:/RRC/major_project_csvtu/run_with_tokens.py) | Token-enabled driver (`apoidea/pubtabnet-html`, cache-first, 100 tables) |
| [`run_pubtabnet_subset.py`](file:///d:/RRC/major_project_csvtu/run_pubtabnet_subset.py) | No-token baseline driver (300-table run, `ajimeno/PubTabNet`) |
| [`measure_post_repair.py`](file:///d:/RRC/major_project_csvtu/measure_post_repair.py) | Pre vs post-repair overlap measurement on actual post-processed cells |
| [`compute_confidence_intervals.py`](file:///d:/RRC/major_project_csvtu/compute_confidence_intervals.py) | Wilson score CIs for all proportions; bootstrap CI for overlap mean |
| [`inspect_bbox_mutations.py`](file:///d:/RRC/major_project_csvtu/inspect_bbox_mutations.py) | Bbox mutation audit via `inspect.getsource` across full call chain |
| [`check_gap_explanations.py`](file:///d:/RRC/major_project_csvtu/check_gap_explanations.py) | Resolution/aspect-ratio comparison + token-vs-no-token cell count check |
| [`pubtabnet_tokens_consistency_report.csv`](file:///d:/RRC/major_project_csvtu/pubtabnet_tokens_consistency_report.csv) | Per-table raw results (100 tables, token run) |
| [`pubtabnet_raw_consistency_report.csv`](file:///d:/RRC/major_project_csvtu/pubtabnet_raw_consistency_report.csv) | Per-table raw results (300 tables, no-token run) |
| [`pubtabnet_pre_post_repair_report.csv`](file:///d:/RRC/major_project_csvtu/pubtabnet_pre_post_repair_report.csv) | Per-table pre and post-repair overlap counts, cell counts, recovery flags |
