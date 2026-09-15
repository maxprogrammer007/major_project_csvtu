# H1 Pilot Study — TableFormer Raw Prediction Consistency
**Date:** 2026-09-15  
**Project:** Major Project CSVTU

---

## What This Folder Contains

A complete, self-contained snapshot of the H1 pilot study measuring how structurally inconsistent TableFormer's raw (pre-repair) predictions are, before `MatchingPostProcessor` runs.

## Files

| File | What it is |
|:---|:---|
| `h1_pilot_findings.md` | **Start here.** Full findings writeup with all key metrics, confidence intervals, bimodal failure structure, IoU-based structural accuracy evaluation vs ground truth, bug-fix history, bbox mutation audit, and scope statement. |
| `check_structural_accuracy.py` | Bipartite greedy IoU matching script (precision, recall, F1, median IoU at IoU@0.5 and IoU@0.75) comparing predicted cell bboxes (raw and post-repair) against PubTabNet ground-truth cell bboxes. |
| `pubtabnet_structural_accuracy_report.csv` | Output of the IoU structural accuracy check across catastrophic and ordinary comparison tables. |
| `pubtabnet_tokens_consistency_report.csv` | Per-table results — 100 tables, real per-cell OCR tokens from `apoidea/pubtabnet-html`. Post-processor gate confirmed open. |
| `pubtabnet_raw_consistency_report.csv` | Per-table results — 300 tables, no-token baseline run (post-processor gate closed). |
| `pubtabnet_pre_post_repair_report.csv` | Per-table pre vs post-repair overlap measurements on actual post-processed cells across all 100 token-enabled tables. |
| `check_ground_truth_cells.py` | Ground-truth cell count extraction streaming from `apoidea/pubtabnet-html`. |
| `compute_confidence_intervals.py` | Wilson score 95% CIs for all proportions + bootstrap CI (B=10,000) for the overlap count mean. Run with `python -X utf8 compute_confidence_intervals.py`. |
| `inspect_bbox_mutations.py` | Source-level bbox mutation audit across CellMatcher + MatchingPostProcessor sub-paths. Run from the project root (requires `docling-ibm-models` on the path). |
| `inspect_639579_overlap.py` | Geometric inspection of the containment overlap introduced by post-processing in table `_639579`. |
| `check_gap_explanations.py` | Empirical checks on candidate explanations for the 40% vs 57% overlap rate gap (table size, token presence, image resolution/aspect ratio). |

## One-Line Summary

> For TableFormer-fast on PubTabNet: raw predictions are **grammatically clean** (100% valid OTSL grids) but **geometrically unreliable** — 57% [47–66%] of tables have overlapping cell bboxes before any repair, with a distinct catastrophic-collapse tail at 5% [2.2–11.2%]. `MatchingPostProcessor` is source-verified as the sole correction mechanism. Ground-truth IoU matching proves that while post-repair achieves 100% precision and recall on clean-count tables (`_670433`), it suffers severe recall collapse (17.6%) on prediction collapse tails (`_629558`), and causes measurable over-pruning (-37% count error, recall dropping from 58.7% to 52.2%) on ordinary overlap tables (`_722429`).

## Next Milestone

SPLERGE (split-and-merge architecture) — same instrumentation, architecturally furthest from autoregressive decoding, maximum contrast value against these TableFormer results.
