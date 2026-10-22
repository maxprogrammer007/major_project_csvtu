"""
E13 — LORE matcher ambiguity verification (OPEN ITEM #2 from the ledger).

The paper claims: "matcher equivalence to official pairTab demonstrated
(0 TP discrepancy; 0/100 ambiguous tables)". This script re-derives the
ambiguity count from cached LORE predictions, on CPU:

  For each table:
    1. Build the IoU matrix (pred boxes x GT boxes) at the paper's matching
       threshold (default 0.5, pairTab convention).
    2. Compute the canonical assignment: max cardinality, then max total IoU
       (Hungarian via scipy.optimize.linear_sum_assignment on a cost matrix
       that first maximizes matches, then weights).
    3. Compute the official-style greedy assignment (first-come first-served
       in prediction order, highest remaining IoU per prediction).
    4. TP discrepancy = TP_hungarian - TP_greedy  (expected 0 everywhere).
    5. Ambiguity probe: for every pair of matched pairs (p1->g1, p2->g2),
       check whether the swapped assignment (p1->g2, p2->g1) is feasible
       (both IoUs >= threshold) AND preserves total IoU within 1e-9.
       Any such swap => the table is AMBIGUOUS (multiple optimal matchings).

Inputs
  --pred-file PATH   JSON of cached predictions:
                     {stem: {"pred_boxes": [[x1,y1,x2,y2],...],
                             "gt_boxes":  [[x1,y1,x2,y2],...]}}
                     Auto-discovery tries: *lore*pred*.json, *lore*cache*.json,
                     scratch/*lore*.json, lore_consistency inputs.
  --iou 0.5          matching threshold

Outputs -> expkit_out/exp13_lore_ambiguity.{json,txt}
Exit code 0 if TP discrepancy == 0 and ambiguous == 0; else 1.

Usage: python expkit/exp13_lore_matcher_ambiguity.py --pred-file <path> [--iou 0.5]
Runtime: seconds (CPU).
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "expkit_out"
OUT.mkdir(exist_ok=True)


def find_pred_file():
    pats = ["*lore*pred*.json", "*lore*cache*.json", "scratch/*lore*pred*.json",
            "scratch/*lore*cache*.json", "scratch/lore*.json", "lore*.json"]
    for p in pats:
        hits = sorted(ROOT.glob(p))
        for h in hits:
            if "sweep" in h.name or "threshold" in h.name or "summary" in h.name:
                continue
            try:
                obj = json.loads(h.read_text(encoding="utf-8"))
                if isinstance(obj, dict) and obj:
                    k = next(iter(obj))
                    if isinstance(obj[k], dict) and any("pred" in key or "box" in key for key in obj[k]):
                        return h
            except Exception:
                continue
    return None


def iou_matrix(preds, gts):
    if len(preds) == 0 or len(gts) == 0:
        return np.zeros((len(preds), len(gts)))
    p = np.asarray(preds, dtype=float)
    g = np.asarray(gts, dtype=float)
    ix1 = np.maximum(p[:, None, 0], g[None, :, 0])
    iy1 = np.maximum(p[:, None, 1], g[None, :, 1])
    ix2 = np.minimum(p[:, None, 2], g[None, :, 2])
    iy2 = np.minimum(p[:, None, 3], g[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    ap = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    union = ap[:, None] + ag[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def hungarian_tp(iou, thr):
    """Max cardinality first, then max total IoU. Returns (tp, matched_pairs)."""
    big = 1e3
    cost = np.where(iou >= thr, big + iou, 0.0)
    r, c = linear_sum_assignment(-cost)
    pairs = [(int(a), int(b)) for a, b in zip(r, c) if iou[a, b] >= thr]
    return len(pairs), pairs


def greedy_tp(iou, thr):
    """First-come (prediction order), best remaining GT; official pairTab style."""
    used_g = set()
    tp = 0
    for i in range(iou.shape[0]):
        best_j, best_v = -1, thr
        for j in range(iou.shape[1]):
            if j in used_g:
                continue
            if iou[i, j] > best_v:
                best_v = iou[i, j]
                best_j = j
        if best_j >= 0:
            used_g.add(best_j)
            tp += 1
    return tp


def ambiguous(iou, thr, pairs, eps=1e-9):
    """2-opt probe for alternative optimal assignments."""
    for a in range(len(pairs)):
        for b in range(a + 1, len(pairs)):
            p1, g1 = pairs[a]
            p2, g2 = pairs[b]
            v_swapped = iou[p1, g2] + iou[p2, g1]
            v_current = iou[p1, g1] + iou[p2, g2]
            if iou[p1, g2] >= thr and iou[p2, g1] >= thr and abs(v_swapped - v_current) <= eps:
                return True, (p1, g1, p2, g2)
    return False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-file", default=None)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    pf = Path(args.pred_file) if args.pred_file else find_pred_file()
    if pf is None or not Path(pf).exists():
        print("[E13] FATAL: cached LORE predictions not found. Pass --pred-file pointing "
              "to the JSON with {stem: {'pred_boxes': [...], 'gt_boxes': [...]}} "
              "(re-dump from measure_lore_consistency.py if needed).")
        sys.exit(2)

    data = json.loads(Path(pf).read_text(encoding="utf-8"))
    print(f"[E13] predictions: {pf} | tables: {len(data)} | IoU threshold: {args.iou}")

    n_disc = 0
    n_amb = 0
    rows = []
    for stem, entry in sorted(data.items()):
        preds = entry.get("pred_boxes") or entry.get("preds") or []
        gts = entry.get("gt_boxes") or entry.get("gts") or []
        iou = iou_matrix(preds, gts)
        tp_h, pairs = hungarian_tp(iou, args.iou)
        tp_g = greedy_tp(iou, args.iou)
        amb, swap = ambiguous(iou, args.iou, pairs)
        n_disc += int(tp_h != tp_g)
        n_amb += int(amb)
        rows.append({"table_id": stem, "tp_hungarian": tp_h, "tp_greedy": tp_g,
                     "ambiguous": bool(amb), "swap": swap})

    disc_tables = [r["table_id"] for r in rows if r["tp_hungarian"] != r["tp_greedy"]]
    amb_tables = [r["table_id"] for r in rows if r["ambiguous"]]

    print(f"\n  tables checked          : {len(rows)}")
    print(f"  TP discrepancy tables   : {n_disc}  {disc_tables[:10]}")
    print(f"  ambiguous tables (2-opt): {n_amb}  {amb_tables[:10]}")
    print(f"\n  Ledger claim '0/100 ambiguous, 0 TP discrepancy' -> "
          f"{'CONFIRMED' if n_disc == 0 and n_amb == 0 else 'REFUTED — update paper'}")

    out = {"pred_file": str(pf), "iou_threshold": args.iou, "n_tables": len(rows),
           "n_tp_discrepancy": n_disc, "n_ambiguous": n_amb,
           "discrepancy_tables": disc_tables, "ambiguous_tables": amb_tables,
           "per_table": rows}
    (OUT / "exp13_lore_ambiguity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[E13] done -> {OUT / 'exp13_lore_ambiguity.json'}")
    sys.exit(0 if n_disc == 0 and n_amb == 0 else 1)


if __name__ == "__main__":
    main()
