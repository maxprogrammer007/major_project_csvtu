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
4. **[`../writeup_2026-09-15/h1_pilot_findings.md`](file:///d:/RRC/major_project_csvtu/writeup_2026-09-15/h1_pilot_findings.md)**:
   - Baseline findings for **Paradigm 1: Sequence Generation (TableFormer)** on PubTabNet.

---

## Status of Four-Paradigm Master Plan

| # | Paradigm | Architecture | Model / Checkpoint | Dataset | Status |
|:---:|:---|:---|:---|:---|:---:|
| 1 | Sequence Generation | Autoregressive OTSL | TableFormer-fast | PubTabNet | **Completed** |
| 2 | Transformer / DETR | Set prediction / Object queries | TATR-v1.1-All | PubTables-1M | **Completed** |
| 3 | Relation Learning | Graph / Logical relation | LORE-TSR (`ckpt_ptn`) | PubTabNet | Up Next |
| 4 | Split-and-Merge | Semantic segmentation | SPLERGE | PubTabNet | Pipeline Build Needed |
