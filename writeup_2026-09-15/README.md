# H1 Pilot Study — TableFormer Raw Prediction Consistency
**Date:** 2026-09-15  
**Project:** Major Project CSVTU

---

## What This Folder Contains

A complete, self-contained snapshot of the H1 pilot study measuring how structurally inconsistent TableFormer's raw (pre-repair) predictions are, before `MatchingPostProcessor` runs.

## Files

| File | What it is |
|:---|:---|
| `h1_pilot_findings.md` | **Start here.** Full findings writeup with all key metrics, confidence intervals, bimodal failure structure, bug-fix history, bbox mutation audit, and scope statement. |
| `pubtabnet_tokens_consistency_report.csv` | Per-table results — 100 tables, real per-cell OCR tokens from `apoidea/pubtabnet-html`. Post-processor gate confirmed open. |
| `pubtabnet_raw_consistency_report.csv` | Per-table results — 300 tables, no-token baseline run (post-processor gate closed). |
| `compute_confidence_intervals.py` | Wilson score 95% CIs for all proportions + bootstrap CI (B=10,000) for the overlap count mean. Run with `python -X utf8 compute_confidence_intervals.py`. |
| `inspect_bbox_mutations.py` | Source-level bbox mutation audit across CellMatcher + MatchingPostProcessor sub-paths. Run from the project root (requires `docling-ibm-models` on the path). |

## One-Line Summary

> For TableFormer-fast on PubTabNet: raw predictions are **grammatically clean** (100% valid OTSL grids) but **geometrically unreliable** — 57% [47–66%] of tables have overlapping cell bboxes before any repair, with a distinct catastrophic-collapse tail at 5% [2.2–11.2%]. `MatchingPostProcessor` is source-verified as the sole correction mechanism.

## Next Milestone

SPLERGE (split-and-merge architecture) — same instrumentation, architecturally furthest from autoregressive decoding, maximum contrast value against these TableFormer results.
