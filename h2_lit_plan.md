# H2 Literature Survey and Implementation Plan

**Context:** H1 is closed across four TSR paradigms (TableFormer, TATR, LORE-TSR, SPLERGE). Every
paradigm showed the same pattern: heuristic post-processing destroys correctly-predicted
structure, not just invalid structure. This document surveys the literature relevant to H2 —
replacing that heuristic repair step with a principled optimization formulation — and lays out a
concretely scoped plan anchored on SPLERGE's `row_merging` failure, the clearest mechanistic
example found in the H1 study.

---

## 1. Literature Survey

### 1.1 TSR-specific work on structural constraints — the closest prior art, and why it isn't the same thing

**TabStruct-Net V2 / TOD-Net** (Visual Understanding of Complex Table Structures from Document
Images, arXiv:2111.07129; extended in *TabStruct-Net V2*, IJDAR 2025) is the single closest piece
of prior art found. It explicitly encodes the same three structural properties this project cares
about — **alignment, continuity, and non-overlap** — as loss terms:

```
L_ol^y = Σ_{i,j} ||(min(y2i, y2j) - max(y1i, y1j))||² · 1(i≠j)
```

optimized via a min-max formulation with trainable weights per structural regularizer. This
confirms the *properties* H2 wants to enforce (no overlap, cell continuity, alignment) are
already recognized in the field as worth optimizing for.

**What's different, and why this doesn't close the gap:** TOD-Net's constraints are **soft,
differentiable training-time regularizers** — penalties baked into the loss function that shape
what the network *tends* to predict, with no hard guarantee at inference time. They also operate
on the network's *own* predictions during training, not on reconciling a **fixed set of already-
made raw predictions** against exact constraints at inference. H1's finding is specifically about
a *separate, later, heuristic post-processing stage* that runs after training is done and
irreversibly discards content. TOD-Net doesn't have — and doesn't try to have — that stage; it
tries to make the initial predictions good enough that heavy post-processing isn't needed. Given
H1 shows raw predictions are already largely correct (TableFormer's raw MedIoU 0.62–0.84; LORE's
raw detections 99% precision; SPLERGE's raw grid 97.6% content-containment) and the *repair step*
is what's destructive, TOD-Net's approach — improve the front end — doesn't address the specific
failure mode this project identified in the back end.

### 1.2 Differentiable and learned NMS / redundancy-resolution — general CV literature, unapplied to TSR

This is a well-established sub-field in general object detection, built on exactly the premise H1
confirms for tables: **heuristic NMS discards true positives.**

- **Learning NMS** (Hosang, Benenson, Schiele — CVPR 2017): a GNN-style network trained to make
  suppression decisions jointly rather than greedily, explicitly motivated by greedy NMS's
  failure in crowded/overlapping scenes.
- **Adaptive NMS** (Liu et al., 2019): a learned per-detection "density" score that raises the
  suppression threshold in crowded regions specifically to stop true positives being suppressed
  — directly analogous to the dense-table cases where SPLERGE's `row_merging` and TATR's row NMS
  destroyed genuine adjacent structures.
- **Differentiable NMS via Sinkhorn Matching** (2025, arXiv:2505.07040): reformulates NMS as a
  differentiable bipartite matching problem solved via Sinkhorn-Knopp, maintaining gradient flow
  and explicitly targeting the "NMS discards true positives, disrupts end-to-end training"
  problem.
- **Quantum-soft QUBO Suppression** (Li & Ghosh): reformulates NMS as a discrete Quadratic
  Unconstrained Binary Optimization problem — closer in spirit to an exact combinatorial
  reformulation than the learned/differentiable approaches above, though the authors report it
  did not outperform classical NMS on standard detection metrics.
- **Optimal-transport-based redundancy resolution** (single-molecule localization microscopy,
  2025): reformulates the training objective as a set-matching problem via an OT loss,
  eliminating NMS at inference entirely.

**The gap:** every one of these methods was developed and evaluated for generic object detection
(pedestrians, cars, 3D LiDAR point clouds, fabric defects, fluorophores) — none for table
structure recognition, and none specifically for the row/column-merge decisions that TableFormer,
TATR, LORE, and SPLERGE all handle as a bespoke heuristic. The technique is proven; the
application to TSR's specific merge/repair problem is not.

### 1.3 Exact and differentiable combinatorial optimization layers (established earlier in this project)

Recapped here because it bears directly on **how** to build the H2 layer, not just **whether**
the gap exists:

- **CombOptNet** (Paulus et al., ICML 2021): integrates an exact ILP solver as a differentiable
  layer, learning both costs and constraints — but requires a *fixed problem template* (constant
  variable/constraint count), confirmed by its own stated limitation and its keypoint-matching
  experiment, which had to force a fixed keypoint count by dropping/subsampling data.
- **LinSATNet** (Wang et al., ICML 2024): generalizes Sinkhorn to arbitrary positive linear
  constraints (not just doubly-stochastic/permutation), demonstrated on routing and partial graph
  matching — but is a continuous relaxation requiring post-hoc rounding, with no guarantee of
  exact feasibility without it.
- **Deep Flow Networks** (Candogan & Foussoul, Chicago Booth WP 26-12): min-cost-flow-based
  function approximators with tractable downstream integer optimization when the learned
  parameter matrix stays near-identity — relevant as the origin point of this whole
  optimization-layer investigation, though (as established at the very start of this project) its
  theoretical machinery doesn't transfer directly to a structured-prediction repair problem.

**Relevance to H2's design choice:** given table sizes in this study are small (tens to a few
hundred candidate cells per table — see the 3×3-to-50×50 rectangle-count analysis from earlier in
this project), an **exact ILP formulation solved directly per table** is likely tractable without
needing a differentiable relaxation layer at all. This matters: it means H2 doesn't need to solve
the harder "differentiable through a combinatorial solver" problem CombOptNet/LinSATNet are built
for — it can call an off-the-shelf solver (e.g., a set-packing/ILP formulation via Gurobi or
OR-Tools) directly at inference time, with no training-time differentiability requirement, since
the goal is to fix a *post-hoc repair step*, not retrain the underlying detection model.

### 1.4 Latent structure / topology inference (established earlier in this project)

Recapped briefly since it's directly relevant to why `row_merging`'s specific failure mode is a
structural, not just parametric, problem: the straddle-rule collapses **regions**, not edges — a
multi-row merge is a higher-order object, not a pairwise relation. Earlier investigation of
latent-graph and latent-topology-inference literature (Neural Relational Inference; "From Latent
Graph to Latent Topology Inference: Differentiable Cell Complex Module," 2023) confirmed this
higher-order-region framing is the right one, but that literature targets *representation
learning* (message passing for downstream prediction), not *exact reconstruction with
content-preservation guarantees* — reinforcing that H2's specific formulation (preserve raw
detections unless a real conflict forces removal) doesn't have a ready-made implementation to
borrow directly; it needs to be built.

---

## 2. The Gap Statement

No existing work combines the three things H2 needs simultaneously:

1. **Applied to TSR's post-hoc repair step specifically** (not generic detection NMS, not
   training-time regularization).
2. **Exact/guaranteed constraint satisfaction at inference** (not a soft loss term, not an
   unrounded continuous relaxation).
3. **An explicit content-preservation objective** — maximizing retained valid raw detections
   subject to consistency constraints, rather than a generic "resolve conflicts" objective that
   treats every discard as equally acceptable.

Framed honestly per the standard the rest of this project has held to: this is a **combination**
gap, not a **primitive** gap — every individual piece (exact solvers, NMS-replacement objectives,
TSR-specific structural constraints) exists separately in the literature. The contribution is
assembling them for this specific, now well-evidenced problem, not inventing new optimization
machinery.

---

## 3. H2 Plan

### 3.1 Target

SPLERGE's `row_merging` straddle-rule (`merge.py`), chosen because it is the most mechanistically
transparent destructive-repair example found across all four paradigms, has an existing
100-table before/after dataset already collected, and its failure condition is a simple, checkable
predicate (`word[3] < row.y1 and word[5] > row.y2`).

### 3.2 Formulation

Replace the unconditional straddle-triggered merge with an exact constrained optimization formulation per table.

#### Formulations: Evolution and Visible Record of Corrections

1. **Initial Single-Variable Formulation (Flawed — retained for transparency):**
   - *Variables:* One binary variable $x_i \in \{0, 1\}$ per raw grid cell indicating whether it is retained.
   - *Why this was flawed:* A single variable per raw cell can only decide whether to *keep* or *drop* an elementary cell; it has no mechanism to represent **merged multi-row or multi-column spans**! If a cell is dropped, what takes its place? If two cells are merged, what represents the new merged rectangle?
   - *Conclusion:* TSR cell assembly over a grid with potential multi-row/col spans is fundamentally a **Set Packing / Exact Partition** problem over a candidate pool of rectangular shapes, not a binary selection over elementary tiles alone.

2. **Corrected Set Packing / ILP Formulation:**
   - **Candidate Pool $\mathcal{C}$:**
     - $\mathcal{C}_{\text{elem}}$: The elementary grid cells formed by the tessellation of row cuts $Y$ and column cuts $X$ (e.g., $28 \times 3 = 84$ tiles for `_554187`).
     - $\mathcal{C}_{\text{merge}}$: Multi-row or multi-column candidate rectangles formed by unioning contiguous elementary cells where an OCR token straddles an internal boundary.
     - Total candidates: $|\mathcal{C}| = |\mathcal{C}_{\text{elem}}| + |\mathcal{C}_{\text{merge}}|$.
   - **Decision Variables:**
     - $x_c \in \{0, 1\}$ for each candidate cell $c \in \mathcal{C}$, where $x_c = 1$ indicates candidate $c$ is active in the final table structure.
   - **Tessellation / Non-Overlap Constraints (Set Packing):**
     - For every elementary grid tile $u = (r, k)$ (row $r$, col $k$), at most one active candidate cell may cover tile $u$:
       $$\sum_{c \in \mathcal{C} : u \in c} x_c \le 1 \quad \forall u \in \mathcal{U}$$
     - In matrix form: $A x \le \mathbf{1}$, where $A_{u, c} = 1$ if candidate $c$ contains elementary tile $u$, else $0$.
   - **Objective:**
     - Maximize total content-weighted value:
       $$\max_{x \in \{0, 1\}^{|\mathcal{C}|}} \sum_{c \in \mathcal{C}} w_c x_c$$
   - **Solver:** Exact ILP via `scipy.optimize.milp` (HiGHS solver) — completely deterministic, no differentiable relaxation needed at inference time.

#### Design Caveats and Failure Guards

> [!CAUTION]
> **Weight Function Caveat ($w_c$): Guarding Against False Merges**
> If $w_c$ were naively set to candidate area or unpenalized token count, a merged span $c_{\text{merge}} = c_1 \cup c_2$ containing two separate tokens (`PCa-004` and `PCa-005`) would receive equal or greater weight than keeping $c_1$ and $c_2$ separate, quietly reproducing SPLERGE's original bug inside the ILP!
> **Requirement:** The weight function $w_c$ must satisfy strict subadditivity for distinct tokens:
> $$w(c_1) + w(c_2) > w(c_1 \cup c_2)$$
> Merging $c_1$ and $c_2$ must only win if an OCR token physically straddles the cut such that neither elementary tile alone contains the token ($\text{containment} < 80\%$), while the merged cell contains it fully ($\text{containment} \ge 80\%$).

> [!NOTE]
> **Solver Runtime Caveat**
> While candidate pool sizes are small for typical tables ($|\mathcal{C}| \sim 80 - 300$), runtime must be measured empirically on actual tables (Milestone 1 on `_554187`) rather than assumed negligible, to confirm HiGHS solves the instance in milliseconds before scaling to the full 100-table suite.

### 3.3 Success criterion

Using the same 100-table sample and instrumentation already in hand:

- **Primary:** recover some fraction of the 634 true positives (IoU ≥ 0.25) destroyed by
  `row_merging`, without reintroducing the overlaps the merge step exists to resolve (i.e.,
  post-optimization overlap rate must remain at or near the 0.0% currently achieved by the
  heuristic).
- **Secondary:** repeat the same measurement on the TableFormer, TATR, and LORE-TSR datasets
  already collected, to test whether a differently-instantiated version of the same formulation
  (content-preservation subject to consistency) generalizes beyond SPLERGE — but only after the
  SPLERGE prototype is validated; this is explicitly a stretch goal, not part of the initial
  scope.

### 3.4 Milestones

1. **Week 1 — Formalize and hand-verify.** Write the ILP formulation precisely; hand-check it on
   2–3 of the already-inspected tables (`_554187`, `_701544`) where the exact raw/post-merge
   cells and GT are already known, to confirm the formulation recovers the expected cells before
   writing a general solver interface.
2. **Week 2 — Build and run on the full sample.** Implement against the existing 100-table
   SPLERGE dataset; reuse `_analyze_raw_cells`-style overlap checking and the token-containment
   audit already built, rather than writing new instrumentation from scratch.
3. **Week 3 — Evaluate and compare.** Report recovered-TP count, overlap rate, and runtime
   against the heuristic baseline, with the same Wilson/bootstrap CI discipline used throughout
   H1. Explicitly test whether runtime is practical at table scale (expected yes, given the
   earlier rectangle-candidate-count analysis, but confirm rather than assume).
4. **Week 4 (stretch) — Cross-paradigm check.** If time permits, adapt the same formulation to
   one other paradigm's repair step (TATR's row NMS is the natural next candidate, given its
   repair mechanism is also geometric and already well understood) to test generalization.

### 3.5 Risks

- **Solver runtime at the dense end of the table-size distribution** (SPLERGE's raw grids reached
  up to several hundred candidate cells) — mitigate by testing runtime early (Milestone 1) rather
  than discovering a scaling problem after full implementation.
- **The token-containment signal may not be available at true inference time** (it currently
  relies on OCR tokens built from ground truth for this study's instrumentation) — for a
  deployment-realistic version, this would need real OCR, following the same reasoning already
  applied when real tokens were built for the TableFormer and SPLERGE H1 measurements.
- **Success may be partial and threshold-dependent** — a result like "recovers 40% of destroyed
  cells but only at IoU ≥ 0.25, not IoU ≥ 0.50" would still be a real, reportable finding, not a
  failure; the success criterion above is intentionally framed as "some fraction," not a specific
  target number, to avoid pressure toward an artificially clean result.

---

## 4. Summary

The literature confirms the properties H2 wants to enforce (non-overlap, continuity, content
preservation) are recognized as valuable across both TSR-specific work (TOD-Net) and general
object detection (learned/differentiable NMS) — but no existing work combines exact,
inference-time, content-preserving optimization applied specifically to TSR's post-hoc repair
stage. That is a real, precisely-scoped gap, not an overclaimed one. The plan above targets the
smallest, best-evidenced instance of that gap first, with success criteria and a fallback
(partial recovery is still a result) defined before implementation begins.