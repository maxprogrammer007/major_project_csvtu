# TSR Cross-Paradigm Project: Master Chat Checkpoint & Handover State
**Repository:** `d:\RRC\major_project_csvtu`  
**Date of Checkpoint:** October 19, 2026 (Local Time: 19:38 IST)  
**Current Git Commit Hash:** `cdcda075c37aaaf1499be92667ea7aeac9a750c3`  
**Purpose:** Comprehensive state preservation document. Whenever you return, this file contains the complete record of discussions, mathematical reconciliations, experimental evidence, paper drafts, and the exact next steps to resume without losing context.

---

## 1. Executive Summary & Gate Status

This project conducts an exhaustive cross-paradigm audit of four representative Table Structure Recognition (TSR) model families:
1. **Autoregressive Sequence Generation:** TableFormer (`docling-models v2.1.0`) on PubTabNet ($N=100$, 5,646 GT cells).
2. **Split-and-Merge Partitioning:** SPLERGE (1D FPN projections) on PubTabNet ($N=100$, 5,646 GT cells).
3. **Object Detection:** TATR-v1.1-All on PubTables-1M ($N=100$, 1,261 GT rows, 576 GT columns).
4. **Relation Learning:** LORE-TSR (CenterNet + DLA-34 + Stacking Transformer) on PubTabNet ($N=100$, 5,646 GT cells).

All audits, empirical reproductions, and statistical reconciliations across all four paradigms are **100% complete and frozen**.

### Gate Status Table

| Gate | Status | Verified Evidence & Provenance |
| :--- | :---: | :--- |
| **LORE Matcher Equivalence** | ✅ **Closed** | Verbatim official `pairTab`, greedy-by-score, and canonical global bipartite matchers yield **100.00% identical pair assignments** across all 100 tables at all thresholds ($\tau=0.35, 0.20, 0.01$). Ambiguity count: **0 out of 100 tables** (0 of 5,646 GT cells) have $\ge 2$ candidate predictions with $\text{IoU} \ge 0.50$. |
| **TableFormer Realistic Headline** | ✅ **Closed** | Realistic OCR word-box pipeline leads: Raw 5,204 $\rightarrow$ Heuristic Snap OFF 4,921 (destroys 283) $\rightarrow$ **MWIS-OCR 5,202** (rescues **+281 cells**, $t = 4.28, p = 4.36 \times 10^{-5}$; Wilcoxon $W = 53.0, p = 1.34 \times 10^{-7}$; 42 wins, 55 ties, 3 losses). Oracle GT token condition (5,203 TPs, +121 rescue) serves as controlled circularity reference (1-cell oracle gap). |
| **H1 Methods Documentation** | ✅ **Closed** | Documented exact 5-step heuristic pipeline from `run_full_metric_suite.py` lines 330–336 (`col_merging` $\rightarrow$ `row_merging` $\rightarrow$ `remove_redundant_seperators(0.2)` $\rightarrow$ `remove_redundant_seperators(0.4)` $\rightarrow$ `merge_header_v2`). |
| **H2_GT vs. H2_OCR Breakdown** | ✅ **Closed** | Re-derived directly from `canonical_greedy_suite_results.csv`: 95 ties ($\Delta = 0$), 3 GT wins (+1, +1, +3), 2 OCR wins (-1, -1), summing to exact net **+3 TPs** ($1,839 - 1,836$). Wilcoxon signed-rank test ranks negative diffs as $2.5 + 2.5 = \mathbf{5.0}$ ($p = 0.4795$). |
| **Checksum Ledger & Supersession** | ✅ **Closed** | Unified 12-artifact frozen ledger with verified SHA-256 hashes and explicit supersession map documenting how early partial files (`tableformer_snapping_full_100.csv`, `tableformer_ocr_control_results.csv`, `tableformer_h2_metrics.csv`, and exploratory $p = 0.257$) were subsumed. |
| **Abstract & §4.2 Realistic Sync** | ✅ **Closed** | Abstract and §4.2 updated in paper draft to lead with realistic OCR pipeline, explain raw continuous regression token independence, and explain the 3 MWIS-OCR losses as density/conflict trade-offs. |
| **Pre-Registration Document Pins** | ✅ **Closed** | [`osf_confirmatory_preregistration_1000_tables.md`](file:///d:/RRC/major_project_csvtu/osf_confirmatory_preregistration_1000_tables.md) rigorously pins sampling frame (`apoidea/pubtabnet-html`, $N=1,000$, seed 20261016, tercile stratification), matcher, evaluator code, canonical baselines, and commit hash `cdcda075c37aaaf1499be92667ea7aeac9a750c3`. |
| **OSF Upload** | ⏳ **Ready** | Upload [`osf_confirmatory_preregistration_1000_tables.md`](file:///d:/RRC/major_project_csvtu/osf_confirmatory_preregistration_1000_tables.md) to OSF / AsPredicted for timestamp. |
| **1,000-Table Confirmatory Run** | ⏳ **Blocked on Upload** | Execute the 1,000-table confirmatory script immediately after pre-registration upload. |

---

## 2. Summary of Findings Across the Four Paradigms

### Paradigm 1: Autoregressive Sequence Generation (TableFormer)
- **Model:** `docling-models v2.1.0` TableFormer-fast on PubTabNet ($N=100$, 5,646 GT cells).
- **The Realistic Pipeline (The Paper's Headline):**
  - Raw Continuous Regressor: **5,204 TPs** (92.17% micro-recall).
  - De-snapped Heuristic (Realistic OCR Word-Boxes): **4,921 TPs** (87.16% micro-recall, destroys 283 valid cells).
  - Exact MWIS Optimization (Realistic OCR Word-Boxes): **5,202 TPs** (92.14% micro-recall).
  - Rescue: MWIS rescues **+281 valid cells** over the de-snapped heuristic ($t = 4.28, p = 4.36 \times 10^{-5}$; Wilcoxon $W = 53.0, p = 1.34 \times 10^{-7}$; 42 wins, 55 ties, 3 losses), closing 99.3% of the heuristic loss.
  - Equivalence to Raw Baseline: Two One-Sided Tests (TOST) confirm equivalence within $\pm 2.0$ TPs ($t_1 = 140.72, t_2 = -143.56, p_{\text{TOST}} = 5.46 \times 10^{-117}$).
  - Token Independence: Raw continuous regression does not consume text tokens; it provides an invariant token-independent benchmark ceiling.
  - The 3 Losses: On 3 of 100 tables, MWIS selected 1 fewer cell due to sub-cell OCR word fragments in dense text blocks creating slight tightness penalty trade-offs between competing boundary candidates.
- **Deconstructing the Published Snapping Gain (Evaluation Circularity):**
  - Published pipeline reports apparent $+284$ gain ($5,204 \rightarrow 5,488$ TPs) via coordinate snapping (`_align_table_cells_to_pdf` step 8a).
  - Under GT tokens (published setup): Snapping to GT token bounding boxes yields **+406 TPs circular inflation** ($5,082 \rightarrow 5,488$).
  - Under realistic OCR word boxes: Snapping to sub-cell word fragments triggers **-1,019 TPs destruction** ($4,921 \rightarrow 3,902$ TPs, $-25.0\%$ below raw).
  - The GT token condition (MWIS achieves 5,203 TPs, +121 rescue) serves as the controlled reference, demonstrating that MWIS is invariant to token geometry (exactly **1-cell gap** between oracle GT tokens and realistic OCR word boxes).

#### Complete 3×2 Factorial Decomposition Table

| Post-Processing Condition | Oracle GT Tokens | Realistic OCR Word-Boxes | Snapping Effect ($\Delta_{\text{snap}}$) |
| :--- | :---: | :---: | :---: |
| **Snapping ENABLED (Heuristic Snap ON)** | **5,488 TPs** (97.20% Micro) | **3,902 TPs** (69.11% Micro) | Token Source Gap: **-1,586 TPs** |
| **Snapping DISABLED (Heuristic Snap OFF)**| **5,082 TPs** (89.99% Micro) | **4,921 TPs** (87.16% Micro) | Token Source Gap: **-161 TPs** |
| **Exact MWIS Optimization (Ours)** | **5,203 TPs** (92.15% Micro) | **5,202 TPs** (92.14% Micro) | Token Source Gap: **-1 TP** |
| **Raw Continuous Regressor Baseline** | **5,204 TPs** (92.17% Micro) | — *(Token-Independent)* | — |
| **Net Snapping Impact** | **+406 TPs** (Circular Inflation) | **-1,019 TPs** (Sub-Cell Mutilation) | **Interaction Delta: 1,425 TPs** |

---

### Paradigm 2: Split-and-Merge Grid Partitioning (SPLERGE)
- **Model:** SPLERGE on PubTabNet ($N=100$, 5,646 GT cells).
- **Destructive Heuristics:**
  - Raw Split Grid: 1,833 TPs (32.47% micro-recall).
  - Heuristic H1 on Geometric Word-Boxes: **1,336 TPs** (23.66% micro-recall, destroys 27.11% of valid cells; degrades 70/100 tables, ties 29, improves 1).
  - Heuristic H1 on Oracle GT Tokens: **1,203 TPs** (21.31% micro-recall, destroys 34.37% of valid cells; degrades 72/100 tables, ties 28, improves 0).
- **Exact Set Packing (H2):**
  - Formulated as Maximum Weight Set Packing solved via branch-and-cut (HiGHS).
  - H2 on Geometric Word-Boxes: **1,836 TPs** (32.52% micro-recall, **+500 net TPs over H1**, $t = 6.66, p = 1.52 \times 10^{-9}$; 71 wins, 29 ties, 0 losses).
  - H2 on Oracle GT Tokens: **1,839 TPs** (32.57% micro-recall; +3 over word-boxes).
- **H2_GT vs. H2_OCR Breakdown:**
  - 95/100 tables identical cell-for-cell ($\Delta = 0$).
  - 3 GT wins: `pubtabnet_tok_566196` (+1), `pubtabnet_tok_723463` (+1), `pubtabnet_tok_578521` (+3).
  - 2 OCR wins: `pubtabnet_tok_642815` (-1), `pubtabnet_tok_665351` (-1).
  - Net aggregate difference: $+3$ TPs. Wilcoxon $W = 5.0, p = 0.4795$. TOST equivalence within $\pm 2.0$ TPs margin: $p_{\text{TOST}} = 5.55 \times 10^{-76}$.
- **Objective Normalization vs. Solver Exactness:**
  - Under unnormalized token-coverage sums ($w_i$), greedy selection collapses to **504 TPs** ($512.6 \pm 9.6$ over 20 randomized seeds) because multi-tile blocks greedily starve valid single-tile cells. Here, exact ILP is indispensable (1,836 vs 504).
  - Under footprint-normalized weights ($w_i / |\text{tiles}_i|$), greedy selection achieves **1,835 TPs** ($1,834.6 \pm 0.8$ over 20 seeds) with a mean Jaccard overlap of **0.9962** (identical in **93 out of 100 tables**).
  - Normalized greedy actively selects 56 valid merge candidates, achieving +2 TPs over the raw split grid (1,833 $\rightarrow$ 1,835).
  - Conclusion: Objective design, not solver capability, is the binding constraint.
- **The Branch B Metric Paradox:**
  - Exact Set Packing achieves massive recall gains (+500 cells), but H1 heuristic scores higher on global tree-edit distance: H1 TEDS-S **0.7318 vs. 0.6725** ($p = 0.0044$) and GriTS-Top **0.7908 vs. 0.7605**.
  - Mechanism: Upstream split models over-segment row bands; heuristic separator pruning deletes empty cuts, creating a shallow HTML tree whose node count matches ground-truth tree depth. Exact optimization correctly preserves granular physical cell detections on over-split rows, which tree-edit metrics heavily penalize.

---

### Paradigm 3: Detection-Based Architectures (TATR)
- **Model:** TATR-v1.1-All on PubTables-1M ($N=100$, 1,261 GT rows, 576 GT columns).
- **Topology Bounds:**
  - Full conflict graph audit across 1,867 candidates (1,286 rows, 581 columns):
    - Isolated Singletons ($K_1$): 1,832 candidates (**98.13%**). Zero competition.
    - Isolated Pairs ($K_2$): 13 components (**0.70%**; 10 row pairs, 3 column pairs). Greedy by score is mathematically identical to MWIS on $K_2$.
    - Overlapping Chains ($P_3$): 3 components (**0.16%**; 2 row chains, 1 column chain).
  - TPs: Raw Rows 1,244, Greedy NMS 1,237, Exact MWIS **1,245 TPs**.
  - Net Gain: Exactness yields exactly +1 row TP across the entire 100-table corpus, strictly isolated to a single row $P_3$ chain on table `PMC6127715_table_0` (where MWIS selected outer rows $A+C$ with weights $2+2=4$ over center row $B$ with weight 3).
  - Finding: Optimization scope is fundamentally bounded by conflict-graph topology; modern DETR backbones rarely emit dense conflict clusters.

---

### Paradigm 4: Relation Learning Architectures (LORE)
- **Model:** LORE-TSR (`ckpt_ptn` reproduction) on PubTabNet ($N=100$, 5,646 GT cells).
- **Upstream Confidence Threshold Artifact:**
  - Default execution at $\tau = 0.35$ reports an apparently dismal **31.19% cell recall** (1,761 TPs) at 99.04% precision. Post-processing NMS is completely dormant.
  - Lowering threshold to $\tau = 0.20$ recovers **90.52% recall** (**5,111 True Positives**) at **99.18% precision** without altering network weights or post-processing logic.
  - Lowering below $\tau < 0.10$ causes a $269\times$ false-positive explosion bounded by CenterNet's fixed $K=100$ candidate query budget (TPs freeze at 5,180 from $\tau=0.10$ down to $\tau=0.01$).
- **Matcher Protocol Equivalence & Degeneracy Verification:**
  - Verbatim official LORE `pairTab` (`AdvancedLiterateMachinery/.../LORE-TSR/src/lib/utils/eval_utils.py`), greedy-by-score, and canonical global bipartite matchers yield **100.00% identical pair assignments** across all 100 tables at all thresholds ($\tau = 0.35, 0.20, 0.01$; checksum `b5ebe3c2...`).
  - Mathematical Mechanism: CenterNet's anchor-free point detection produces non-overlapping local maxima.
  - Ambiguity Count: In the 100-table benchmark (5,646 GT cells), the count of tables with any $\ge 2$-candidate assignment ambiguity is **0 out of 100 tables** (0 of 5,646 GT cells) at both operational thresholds ($\tau = 0.35$ and $\tau = 0.20$). Because no GT cell has multiple competing predictions, sequential first-match, score-sorted greedy, and global bipartite matching are mathematically degenerate and yield strictly identical match assignments.

---

## 3. Authoritative Provenance Ledger & Checksum Registry

Every reported number is traceable to frozen canonical data artifacts with verifiable SHA-256 checksums:

| Artifact Path | Description | SHA-256 Checksum |
| :--- | :--- | :--- |
| `canonical_greedy_suite_results.csv` | SPLERGE Greedy & ILP 20-seed runs across all 100 tables | `1d54969b759a9bfcc646a25e8a616b007d2e10bb6952253a8f428cf21169ba07` |
| `canonical_greedy_suite_summary.json` | SPLERGE Greedy & ILP summary statistics & Jaccard overlap | `f9a85caff9d81588b0c559f51c57cacc8af290ca68f01bf4ad2e608acb94313e` |
| `metric_suite_results.csv` | Multi-metric suite across token sources (H1, H2, Raw) | `0a5c8d3a8244131deeb3e5871981339bbf1d23d9cf0348617e15bb59d9cb083c` |
| `metric_suite_summary.json` | Multi-metric summary statistics & degradation counts | `56c727424a627d3347a2f3bc75527ee7dd5f55df9d907fa26630e923b2115620` |
| `tatr_pubtables1m_consistency_report.csv` | TATR row/column candidates & conflict topology | `ccf7908d2a94f4b652e7202974ac5543fd998caa8b012d42636939b5af4da31e` |
| `lore_threshold_sweep_results.csv` | LORE 100-table confidence sweep ($\tau=0.35 \rightarrow 0.01$) | `4bd31fec232b7c7b6d2cce48107dd192ed997345c6ea672060d52e772530460f` |
| `lore_threshold_sweep_summary.json` | LORE 100-table sweep summary statistics & precision/recall | `406c11b0593b4f62ca531dbbb543dd59d280d5ee82650ba2d1007a82c4ff2b70` |
| `scratch/lore_matcher_comparison_results.json` | LORE `pairTab` vs. Bipartite 100% equivalence proof | `b5ebe3c25f948db06722eccbd303bda8c5293e58da5e858b01cbcc6ecefdb903` |
| `scratch/tableformer_2x2_factorial_results.csv` | TableFormer 2×2 factorial evaluation (Tokens × Snapping) | `ae76718c9f629302ee0ed9f6e607c5751a2d719080cebec53dff384fbd367b6e` |
| `scratch/tableformer_2x2_factorial_summary.json` | TableFormer 2×2 factorial summary statistics & decomposition | `be8a609cb20ccb9e4c71928b51df3e6f03b3074b6da223cec5796aa61c9d39ea` |
| `scratch/tableformer_mwis_factorial_results.csv` | TableFormer MWIS GT vs. OCR word-box evaluation | `195f3935d94aa48adbcc62a6da1ceb2677d135620b86452de1fdaa630500c118` |
| `scratch/tableformer_mwis_factorial_summary.json` | TableFormer MWIS summary statistics & TOST equivalence | `ec2c9b99c7ba2fe9ddff439374d127d8445314a20c27e080e1de64a2c8087207` |

### Explicit Artifact Supersession Map

| Superseded Artifact | SHA-256 Checksum | Scope & Reason for Supersession | Authoritative Frozen Successor |
| :--- | :--- | :--- | :--- |
| `scratch/tableformer_snapping_full_100.csv` | `ce56abc1...` | Preliminary 1-column GT snapping trial (Raw 5,204 vs. GT Snap ON 5,488 vs. GT Snap OFF 5,082). Lacked OCR word-box controls. | `scratch/tableformer_2x2_factorial_results.csv` |
| `scratch/tableformer_ocr_control_results.csv` | `e7fdd758...` | Standalone single-column OCR control run (OCR Snap ON 3,902). Integrated into unified factorial matrix. | `scratch/tableformer_2x2_factorial_results.csv` |
| `tableformer_h2_metrics.csv` | `03b1738d...` | Preliminary MWIS trial evaluated exclusively on GT tokens (5,203 TPs). Lacked OCR word-box density weights. | `scratch/tableformer_mwis_factorial_results.csv` |
| Early Exploratory Script Outputs (`optimize_splerge_consistency.py`) | *N/A (untracked script)* | Reported H2-GT as 1,836 TPs due to an unconstrained first-come-first-served prediction loop (`for pb in pred_boxes:`). | `canonical_greedy_suite_results.csv` (1,839 GT / 1,836 OCR TPs) |
| Early exploratory $p = 0.257$ statistic | *N/A (pre-canonical draft)* | Derived from exploratory unnormalized split runs without verified bipartite tie-breaking. | Canonical Paired Test ($t = 0.8308, p = 0.4081$; $W = 5.0, p = 0.4795$; TOST $p = 5.55 \times 10^{-76}$) |

---

## 4. Key Document Locations

All major writeups and registration documents have been synchronized and stored in two locations (both in workspace and brain artifacts):

1. **Pre-Registration Document:**
   - Workspace Path: [`osf_confirmatory_preregistration_1000_tables.md`](file:///d:/RRC/major_project_csvtu/osf_confirmatory_preregistration_1000_tables.md)
   - Brain Artifact: `C:\Users\abhin\.gemini\antigravity-ide\brain\2184674e-262a-43c3-a745-2a2183147167\osf_confirmatory_preregistration_1000_tables.md`
2. **Master Architecture & Provenance Markdown:**
   - Workspace Path: [`writeup_2026-09-16/four_paradigm_master_table.md`](file:///d:/RRC/major_project_csvtu/writeup_2026-09-16/four_paradigm_master_table.md)
3. **Paper Draft (Abstract, Contributions, & Section 4):**
   - Workspace Path: [`writeup_2026-09-16/reframed_paper_draft_section4.md`](file:///d:/RRC/major_project_csvtu/writeup_2026-09-16/reframed_paper_draft_section4.md)
   - Brain Artifact: `C:\Users\abhin\.gemini\antigravity-ide\brain\2184674e-262a-43c3-a745-2a2183147167\reframed_paper_draft_section4.md`

---

## 5. Exact Next Steps to Resume

When you return, here is the exact execution sequence to follow:

### Step 1: Upload the Pre-Registration Document (User Action)
- Open [`osf_confirmatory_preregistration_1000_tables.md`](file:///d:/RRC/major_project_csvtu/osf_confirmatory_preregistration_1000_tables.md).
- Copy its contents and create a registration on [Open Science Framework (OSF)](https://osf.io/) or [AsPredicted](https://aspredicted.org/).
- Note the public registration URL and timestamp (securing the prospective registration timestamp strictly before launching the 1,000-table inference).

### Step 2: Launch the 1,000-Table Scale Study
Once the pre-registration timestamp is secured:
- Sample $N = 1,000$ tables from `apoidea/pubtabnet-html` (`test` / `validation` splits) using random seed `20261016`, stratified across spanning fraction terciles:
  - Low (< 10% spanning): 350 tables
  - Medium (10% - 25% spanning): 350 tables
  - High (> 25% spanning): 300 tables
- Run the confirmatory evaluation script to evaluate:
  - $H_{\text{scale}, 1a}$: SPLERGE H2 (Exact ILP) vs. H1 (Heuristic) on geometric word-boxes (testing the $+500$ TP win replication, $p < 0.001$).
  - $H_{\text{scale}, 1b}$: Branch B metric divergence replication (testing H1 TEDS-S $> 0.70$ vs H2 $< 0.68$).
  - $H_{\text{scale}, 2}$: TableFormer 2×2 factorial and MWIS OCR rescue (testing $+281$ rescue and TOST equivalence within $\pm 2.0$ TPs).
  - $H_{\text{scale}, 3}$: Normalized greedy vs. ILP Jaccard similarity ($\ge 0.98$) and unnormalized greedy collapse ($< 50\%$).

### Step 3: Paper Finalization
- Inject the 1,000-table confirmatory metrics into Section 4 and Table 1.
- All baseline numbers, mathematical mechanisms, and provenance exhibits are already written and verified.
