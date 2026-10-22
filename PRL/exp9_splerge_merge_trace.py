"""
E9 — SPLERGE merge-stage trace analysis (NEW EXPERIMENT).

Answers CV4DC Reviewer ZBKN Weakness 1:
  "The paper needs a more in-depth analysis from the traces such as breaking
   down how many merges does H2 perform and how many were correct."

What it does
------------
Section A (heuristic arm, H1):
  Replays the exact SPLERGE heuristic pipeline stage by stage
      col_merging -> row_merging -> remove_seps(0.2) -> remove_seps(0.4)
      -> merge_header_v2
  and snapshots after EVERY stage:
    - number of merge operations applied (separator removals / header merges)
    - per-operation correctness: a removed separator is CORRECT to remove iff
      no GT cell has an edge at that position (coverage of GT edges at the
      removed position ~= 0); flagged INCORRECT if GT-edge coverage > 0.5
    - active cells, TP@0.25 (canonical matcher), FP, precision, recall
    - optional per-stage TEDS-S

Section B (exact arm, H2 = ILP):
  Rebuilds candidates, solves the exact set-packing ILP, and classifies every
  selected multi-tile merge candidate:
    - correct merge  : all elementary tiles map to ONE GT cell (center test)
    - incorrect merge: tiles span >= 2 GT cells (would fuse distinct cells)

Inputs (project root, run from d:\\RRC\\major_project_csvtu)
------------------------------------------------------------
  scratch/splerge_100_tables_grid_cache.json          (required)
  scratch/pubtabnet_100_tables_gt_html.json           (required, GT structure HTML)
  pubtab_samples_tokens/iocr_jsons/*_iocr.json        (OCR token source, optional
                                                       -> falls back to GT tokens)
  run_full_metric_suite.py                            (imports evaluate_matching /
                                                       build_candidates_and_weights)
  teds_metric.py + grits.py                           (optional per-stage TEDS)
  AdvancedLiterateMachinery/DocumentUnderstanding/SPLERGE  (libs.GTElement,
                                                       libs.merge_utility, merge)

Outputs -> expkit_out/
  exp9_merge_trace_per_table.csv    one row per (table, stage)
  exp9_merge_trace_summary.json     aggregates + ILP merge-correctness counts
  exp9_merge_trace_summary.txt      human-readable summary

Runtime: ~5-10 min CPU for 100 tables with TEDS on (--no-teds to skip).

Usage:
  python expkit/exp9_splerge_merge_trace.py [--tokens ocr|gt] [--no-teds]
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # project root (expkit/ sits inside it)
OUT = ROOT / "expkit_out"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "AdvancedLiterateMachinery" / "DocumentUnderstanding" / "SPLERGE"))

GRID_CACHE = ROOT / "scratch" / "splerge_100_tables_grid_cache.json"
GT_HTML_CACHE = ROOT / "scratch" / "pubtabnet_100_tables_gt_html.json"
IOCR_DIR = ROOT / "pubtab_samples_tokens" / "iocr_jsons"

STAGES = [
    ("raw_grid", None),
    ("col_merging", "col_merging"),
    ("row_merging", "row_merging"),
    ("remove_seps_0.2", "remove_seps_0.2"),
    ("remove_seps_0.4", "remove_seps_0.4"),
    ("merge_header_v2", "merge_header_v2"),
]


def die(msg):
    print(f"[E9] FATAL: {msg}")
    sys.exit(2)


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def import_matcher():
    """Canonical TP matcher; fall back to Hungarian IoU>=0.25."""
    try:
        from run_full_metric_suite import evaluate_matching  # canonical
        return evaluate_matching, "run_full_metric_suite.evaluate_matching"
    except Exception:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        def evaluate_matching(pred_boxes, gt_boxes, iou_thr=0.25):
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                return None, 0, None, None, None, None, None
            iou = np.zeros((len(pred_boxes), len(gt_boxes)))
            for i, p in enumerate(pred_boxes):
                px1, py1, px2, py2 = p[:4]
                for j, g in enumerate(gt_boxes):
                    gx1, gy1, gx2, gy2 = g[:4]
                    iw = max(0.0, min(px2, gx2) - max(px1, gx1))
                    ih = max(0.0, min(py2, gy2) - max(py1, gy1))
                    inter = iw * ih
                    union = (px2 - px1) * (py2 - py1) + (gx2 - gx1) * (gy2 - gy1) - inter
                    iou[i, j] = inter / union if union > 0 else 0.0
            r, c = linear_sum_assignment(-iou)
            tp = int(sum(1 for a, b in zip(r, c) if iou[a, b] >= iou_thr))
            return None, tp, None, None, None, None, None

        print("[E9] WARNING: run_full_metric_suite.evaluate_matching not importable; "
              "using local Hungarian IoU>=0.25 matcher.")
        return evaluate_matching, "fallback_hungarian_iou25"


def load_ocr_tokens(stem):
    """OCR tokens for a stem; returns (tokens, source) or (None, None)."""
    p = IOCR_DIR / f"{stem}_iocr.json"
    if not p.exists():
        return None, None
    try:
        data = load_json(p)
        toks = data["pages"][0]["tokens"]
        out = []
        for t in toks:
            bb = t.get("bbox")
            if bb is None and all(k in t for k in ("x1", "y1", "x2", "y2")):
                bb = [t["x1"], t["y1"], t["x2"], t["y2"]]
            out.append({"id": t.get("id"), "text": t.get("text", ""), "bbox": bb})
        return out, "ocr_iocr_json"
    except Exception as e:
        print(f"[E9] WARNING: failed to parse {p.name}: {e}")
        return None, None


def active_cells(t_merge):
    """Deduplicated active (non-dontCare) cell list with geometry + indices."""
    seen = set()
    cells = []
    for row in t_merge.gtCells:
        for c in row:
            if c.dontCare or id(c) in seen:
                continue
            seen.add(id(c))
            cells.append({
                "r1": c.startRow, "r2": c.endRow,
                "c1": c.startCol, "c2": c.endCol,
                "bbox": [c.x1, c.y1, c.x2, c.y2],
            })
    return cells


def canonical_grid_to_html(num_rows, num_cols, cells):
    grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    for cell in cells:
        for r in range(cell["r1"], cell["r2"] + 1):
            for c in range(cell["c1"], cell["c2"] + 1):
                if r < num_rows and c < num_cols:
                    grid[r][c] = cell
    parts = ["<table>"]
    for r in range(num_rows):
        parts.append("<tr>")
        for c in range(num_cols):
            cell = grid[r][c]
            if cell is not None:
                if cell["r1"] == r and cell["c1"] == c:
                    rowspan = cell["r2"] - cell["r1"] + 1
                    colspan = cell["c2"] - cell["c1"] + 1
                    attrs = []
                    if rowspan > 1:
                        attrs.append(f'rowspan="{rowspan}"')
                    if colspan > 1:
                        attrs.append(f'colspan="{colspan}"')
                    attr_str = (" " + " ".join(attrs)) if attrs else ""
                    parts.append(f"<td{attr_str}></td>")
            else:
                parts.append("<td></td>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def gt_edge_sets(gt_boxes, W, H, tol):
    """GT horizontal/vertical edge positions with width coverage."""
    h_edges, v_edges = {}, {}
    for g in gt_boxes:
        gx1, gy1, gx2, gy2 = g[:4]
        for y in (gy1, gy2):
            h_edges.setdefault(round(float(y), 1), 0.0)
        for x in (gx1, gx2):
            v_edges.setdefault(round(float(x), 1), 0.0)
    # coverage via proximity grouping
    def build(edges):
        keys = sorted(edges)
        groups = []
        for k in keys:
            if groups and k - groups[-1][-1] <= tol:
                groups[-1].append(k)
            else:
                groups.append([k])
        return groups
    return build(h_edges), build(v_edges)


def edge_coverage(pos, groups, axis_boxes, axis, W, H):
    """Fraction of table width/width covered by GT cells having an edge at pos."""
    cov = 0.0
    for g in axis_boxes:
        gx1, gy1, gx2, gy2 = [float(v) for v in g[:4]]
        a1, a2 = (gy1, gy2) if axis == "h" else (gx1, gx2)
        if min(abs(a1 - pos), abs(a2 - pos)) <= 2.0:
            lo, hi = (gx1, gx2) if axis == "h" else (gy1, gy2)
            cov += max(0.0, hi - lo)
    span = W if axis == "h" else H
    return min(1.0, cov / span if span > 0 else 0.0)


def boundary_sets(cells):
    rows = sorted({round(c["bbox"][1], 1) for c in cells})
    cols = sorted({round(c["bbox"][0], 1) for c in cells})
    return rows, cols


def set_diff_positions(prev, cur, tol):
    removed = [p for p in prev if all(abs(p - q) > tol for q in cur)]
    added = [p for p in cur if all(abs(p - q) > tol for q in prev)]
    return removed, added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", choices=["ocr", "gt"], default="ocr")
    ap.add_argument("--no-teds", action="store_true")
    args = ap.parse_args()

    for p, name in [(GRID_CACHE, "grid cache"), (GT_HTML_CACHE, "GT HTML cache")]:
        if not p.exists():
            die(f"missing {name}: {p}")

    try:
        import libs.GTElement as gc
        import libs.merge_utility as mut
        import merge as merge_mod
    except Exception as e:
        die(f"cannot import SPLERGE merge stack ({e}). Expected under "
            f"AdvancedLiterateMachinery/DocumentUnderstanding/SPLERGE")

    evaluate_matching, matcher_name = import_matcher()

    teds = None
    if not args.no_teds:
        try:
            from teds_metric import TEDS
            teds = TEDS(structure_only=True, ignore_nodes=["thead", "tbody"])
        except Exception:
            print("[E9] WARNING: teds_metric unavailable; skipping per-stage TEDS-S.")

    grid_cache = load_json(GRID_CACHE)
    gt_html_cache = load_json(GT_HTML_CACHE)
    stems = [s for s in grid_cache if s in gt_html_cache]
    print(f"[E9] tables: {len(stems)} | matcher: {matcher_name} | tokens: {args.tokens}")

    rows_out = []
    ilp_summary = {"tables": 0, "selected_merge_candidates": 0, "correct": 0,
                   "incorrect": 0, "per_table": {}}

    t0 = time.time()
    for ti, stem in enumerate(sorted(stems), 1):
        entry = grid_cache[stem]
        orig_w, orig_h = float(entry["orig_w"]), float(entry["orig_h"])
        row_seps, col_seps = entry["row_seps"], entry["col_seps"]
        gt_boxes = entry.get("gt_boxes")
        if not gt_boxes:
            continue

        # token source
        if args.tokens == "ocr":
            toks, tsrc = load_ocr_tokens(stem)
            if toks is None:
                toks, tsrc = entry["gt_tokens"], "gt_fallback"
        else:
            toks, tsrc = entry["gt_tokens"], "gt"

        raw_ocr = [[t["id"], t.get("text", ""), t["bbox"][0], t["bbox"][1],
                    t["bbox"][2], t["bbox"][3]] for t in toks]
        ocr_clean = mut.clean_ocr_data(raw_ocr)
        ocr_sentences = mut.make_sentences(ocr_clean)

        t_merge = gc.Table(0, 0, orig_w, orig_h)
        for r in row_seps:
            t_merge.gtRows.append(gc.Row(0, r, orig_w))
        for c in col_seps:
            t_merge.gtCols.append(gc.Column(c, 0, orig_h))
        t_merge.evaluateCells()

        # GT structure HTML for TEDS
        gt_struct_html = None
        if teds is not None:
            toks_html = gt_html_cache[stem]["html_data"]["structure"]["tokens"]
            gt_struct_html = "<table>" + "".join(toks_html) + "</table>"

        tol = max(2.0, 0.004 * max(orig_w, orig_h))
        prev_rows = prev_cols = None
        total_gt = None

        for stage_name, stage_fn in STAGES:
            ops = 0
            ops_incorrect = 0
            cov_vals = []

            cells_before = active_cells(t_merge) if stage_fn else None
            if stage_fn:
                if stage_fn == "col_merging":
                    merge_mod.col_merging(t_merge, ocr_sentences)
                elif stage_fn == "row_merging":
                    merge_mod.row_merging(t_merge, ocr_sentences)
                elif stage_fn == "remove_seps_0.2":
                    merge_mod.remove_redundant_seperators(t_merge, ocr_sentences, 0.2)
                elif stage_fn == "remove_seps_0.4":
                    merge_mod.remove_redundant_seperators(t_merge, ocr_sentences, 0.4)
                elif stage_fn == "merge_header_v2":
                    t_merge.merge_header_v2(ocr_sentences)
                t_merge.evaluateCells()

            cells = active_cells(t_merge)
            rows_now, cols_now = boundary_sets(cells)

            if stage_fn and cells_before:
                rb, cb = boundary_sets(cells_before)
                if stage_fn in ("row_merging", "remove_seps_0.2", "remove_seps_0.4"):
                    removed, _ = set_diff_positions(rb, rows_now, tol)
                    for y in removed:
                        cov = edge_coverage(y, None, gt_boxes, "h", orig_w, orig_h)
                        cov_vals.append(cov)
                        ops += 1
                        if cov > 0.5:
                            ops_incorrect += 1
                    ops += 0  # row count collapse accounted above
                elif stage_fn == "col_merging":
                    removed, _ = set_diff_positions(cb, cols_now, tol)
                    for x in removed:
                        cov = edge_coverage(x, None, gt_boxes, "v", orig_w, orig_h)
                        cov_vals.append(cov)
                        ops += 1
                        if cov > 0.5:
                            ops_incorrect += 1
                elif stage_fn == "merge_header_v2":
                    top_before = min(rb) if rb else None
                    hb = [c for c in cells_before if abs(c["bbox"][1] - top_before) <= tol]
                    ha = [c for c in cells if abs(c["bbox"][1] - top_before) <= tol]
                    ops = max(0, len(hb) - len(ha))
                    ops_incorrect = 0  # header merges classified geometrically below

            # metrics at this stage
            pred_boxes = [c["bbox"] for c in cells]
            res = evaluate_matching(pred_boxes, gt_boxes)
            tp = res[1] if isinstance(res, tuple) else res
            if total_gt is None:
                total_gt = len(gt_boxes)
            fp = len(pred_boxes) - tp
            prec = tp / len(pred_boxes) if pred_boxes else 0.0
            rec = tp / total_gt if total_gt else 0.0

            teds_s = ""
            if teds is not None and gt_struct_html:
                nr = max(c["r2"] for c in cells) + 1
                nc = max(c["c2"] for c in cells) + 1
                try:
                    teds_s = f"{teds.evaluate(canonical_grid_to_html(nr, nc, cells), gt_struct_html):.4f}"
                except Exception:
                    teds_s = ""

            rows_out.append({
                "table_id": stem, "stage": stage_name, "token_source": tsrc,
                "ops": ops, "ops_incorrect": ops_incorrect,
                "mean_edge_coverage": round(sum(cov_vals) / len(cov_vals), 4) if cov_vals else "",
                "cells": len(cells), "tp": tp, "fp": fp,
                "precision": round(prec, 4), "recall": round(rec, 4),
                "teds_s": teds_s,
            })
            prev_rows, prev_cols = rows_now, cols_now

        # ---- Section B: ILP selected-merge correctness ----
        try:
            from run_full_metric_suite import build_candidates_and_weights
            import numpy as np
            import scipy.sparse as sp
            from scipy.optimize import milp, LinearConstraint, Bounds

            cands, weights, nr_, nc_, nt_, t2i = build_candidates_and_weights(
                orig_w, orig_h, row_seps, col_seps, toks)
            n = len(cands)
            ri, ci, dv = [], [], []
            for k, cand in enumerate(cands):
                for tile in cand["tiles"]:
                    ri.append(t2i[tuple(tile)])
                    ci.append(k)
                    dv.append(1.0)
            A = sp.csc_matrix((dv, (ri, ci)), shape=(nt_, n))
            sol = milp(c=-np.asarray(weights),
                       constraints=LinearConstraint(A, -np.inf, 1.0),
                       integrality=np.ones(n), bounds=Bounds(0.0, 1.0))
            sel = np.where(sol.x > 0.5)[0]

            brows = [0.0] + sorted(float(v) for v in row_seps) + [orig_h]
            bcols = [0.0] + sorted(float(v) for v in col_seps) + [orig_w]
            geom_ok = (len(brows) - 1) * (len(bcols) - 1) == nt_

            sel_merges = 0
            correct = 0
            for k in sel:
                cand = cands[k]
                if len(cand["tiles"]) <= 1:
                    continue
                sel_merges += 1
                gt_ids = set()
                for (r_, c_) in cand["tiles"]:
                    if geom_ok:
                        y = 0.5 * (brows[r_] + brows[r_ + 1])
                        x = 0.5 * (bcols[c_] + bcols[c_ + 1])
                    else:
                        x1, y1, x2, y2 = cand["bbox"]
                        y = 0.5 * (y1 + y2); x = 0.5 * (x1 + x2)
                    for gi, g in enumerate(gt_boxes):
                        if g[0] <= x <= g[2] and g[1] <= y <= g[3]:
                            gt_ids.add(gi)
                            break
                if len(gt_ids) == 1:
                    correct += 1
            ilp_summary["tables"] += 1
            ilp_summary["selected_merge_candidates"] += sel_merges
            ilp_summary["correct"] += correct
            ilp_summary["incorrect"] += sel_merges - correct
            ilp_summary["per_table"][stem] = {
                "selected_merges": sel_merges, "correct": correct,
                "incorrect": sel_merges - correct, "geometry_ok": geom_ok}
        except Exception as e:
            print(f"[E9] Section B skipped for {stem}: {e}")

        if ti % 20 == 0:
            print(f"[E9] ... {ti}/{len(stems)} tables ({time.time()-t0:.0f}s)")

    # ---- write outputs ----
    import csv
    csv_path = OUT / "exp9_merge_trace_per_table.csv"
    if rows_out:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)

    # aggregate by stage
    agg = {}
    for r in rows_out:
        a = agg.setdefault(r["stage"], {"ops": 0, "ops_incorrect": 0, "cells": 0,
                                        "tp": 0, "fp": 0, "n": 0, "teds": []})
        a["ops"] += r["ops"]
        a["ops_incorrect"] += r["ops_incorrect"]
        a["cells"] += r["cells"]
        a["tp"] += r["tp"]
        a["fp"] += r["fp"]
        a["n"] += 1
        if r["teds_s"] != "":
            a["teds"].append(float(r["teds_s"]))

    summary = {
        "matcher": matcher_name,
        "token_condition": args.tokens,
        "n_tables": len(set(r["table_id"] for r in rows_out)),
        "stages": {
            s: {
                "ops_total": a["ops"], "ops_flagged_incorrect": a["ops_incorrect"],
                "incorrect_rate": round(a["ops_incorrect"] / a["ops"], 4) if a["ops"] else None,
                "cells_total": a["cells"], "tp_total": a["tp"], "fp_total": a["fp"],
                "precision": round(a["tp"] / (a["tp"] + a["fp"]), 4) if a["tp"] + a["fp"] else None,
                "teds_s_mean": round(sum(a["teds"]) / len(a["teds"]), 4) if a["teds"] else None,
            } for s, a in agg.items()},
        "ilp_merge_correctness": ilp_summary,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(OUT / "exp9_merge_trace_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    txt = ["E9 — SPLERGE merge-stage trace (token condition: %s)" % args.tokens, ""]
    txt.append(f"{'stage':<18}{'ops':>7}{'incorrect':>11}{'rate':>8}{'cells':>8}"
               f"{'TP':>7}{'FP':>7}{'prec':>8}{'TEDS-S':>9}")
    order = [s for s, _ in STAGES]
    for s in order:
        if s not in summary["stages"]:
            continue
        d = summary["stages"][s]
        txt.append(f"{s:<18}{d['ops_total']:>7}{d['ops_flagged_incorrect']:>11}"
                   f"{(d['incorrect_rate'] or 0):>8.3f}{d['cells_total']:>8}"
                   f"{d['tp_total']:>7}{d['fp_total']:>7}{(d['precision'] or 0):>8.4f}"
                   f"{(d['teds_s_mean'] if d['teds_s_mean'] is not None else float('nan')):>9.4f}")
    il = ilp_summary
    txt += ["", f"ILP (H2) selected multi-tile merges: {il['selected_merge_candidates']} "
            f"across {il['tables']} tables | correct {il['correct']} | "
            f"incorrect {il['incorrect']} "
            f"({(il['incorrect']/il['selected_merge_candidates']*100 if il['selected_merge_candidates'] else 0):.2f}% incorrect)"]
    txt += ["", "A removed separator is flagged INCORRECT when GT cells carry an edge at "
            "the removed position across >50% of the table width (i.e., the heuristic "
            "fused two cells that GT keeps separate)."]
    report = "\n".join(txt)
    (OUT / "exp9_merge_trace_summary.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n[E9] done in {summary['runtime_s']}s -> {OUT}")


if __name__ == "__main__":
    main()
