from pathlib import Path

target = Path("PRL/exp12_bridge_validation.py")
lines = target.read_text(encoding="utf-8").splitlines()
start = lines.index("def recompute_splerge():")
end = lines.index("def main():")

new_func = """def recompute_splerge():
    \"\"\"Independent recompute of raw / H1 / H2-ILP / greedy from caches.\"\"\"
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "AdvancedLiterateMachinery" / "DocumentUnderstanding" / "SPLERGE"))
    grid_cache = json.loads((ROOT / "scratch" / "splerge_100_tables_grid_cache.json").read_text())
    try:
        from run_full_metric_suite import (
            detect_ocr_tokens, run_splerge_heuristic,
            build_candidates_and_weights, evaluate_matching,
            solve_milp, solve_greedy
        )
    except Exception as e:
        P(f"  [warn] cannot import run_full_metric_suite ({e}); SPLERGE recompute skipped")
        return None

    tot = {"gt": 0, "raw": 0, "h1_ocr": 0, "h2_ocr": 0, "greedy_std": 0, "greedy_norm": 0,
           "h1_gt": 0, "h2_gt": 0}
    t0 = time.time()
    img_dir = ROOT / "pubtab_samples_tokens" / "images"

    for i, (stem, entry) in enumerate(sorted(grid_cache.items()), 1):
        orig_w, orig_h = float(entry["orig_w"]), float(entry["orig_h"])
        row_seps, col_seps = entry["row_seps"], entry["col_seps"]
        gt_boxes = entry["gt_boxes"]
        gt_toks = entry["gt_tokens"]
        tot["gt"] += len(gt_boxes)

        img_path = img_dir / f"{stem}.png"
        ocr_toks = detect_ocr_tokens(img_path) if img_path.exists() else gt_toks

        def count_tp(boxes):
            return evaluate_matching(boxes, gt_boxes)[1]

        # raw grid
        brows = [0.0] + sorted(float(v) for v in row_seps) + [orig_h]
        bcols = [0.0] + sorted(float(v) for v in col_seps) + [orig_w]
        raw_boxes = []
        for r in range(len(brows) - 1):
            for c in range(len(bcols) - 1):
                raw_boxes.append([bcols[c], brows[r], bcols[c + 1], brows[r + 1]])
        tot["raw"] += count_tp(raw_boxes)

        # H1 heuristic (OCR tokens)
        h1ocr_cells, _, _ = run_splerge_heuristic(orig_w, orig_h, row_seps, col_seps, ocr_toks)
        tot["h1_ocr"] += count_tp([c["bbox"] for c in h1ocr_cells])

        # H1 heuristic (GT tokens)
        h1gt_cells, _, _ = run_splerge_heuristic(orig_w, orig_h, row_seps, col_seps, gt_toks)
        tot["h1_gt"] += count_tp([c["bbox"] for c in h1gt_cells])

        # candidates + ILP + greedy (OCR condition)
        cands, weights, nr_, nc_, nt_, t2i = build_candidates_and_weights(
            orig_w, orig_h, row_seps, col_seps, ocr_toks)
        
        sel_h2, _ = solve_milp(cands, weights, nt_, t2i)
        tot["h2_ocr"] += count_tp([c["bbox"] for c in sel_h2])

        sel_std = solve_greedy(cands, weights, seed=42)
        tot["greedy_std"] += count_tp([c["bbox"] for c in sel_std])

        n = len(cands)
        norm_w = [weights[k] / len(cands[k]["tiles"]) for k in range(n)]
        order_norm = sorted(range(n), key=lambda k: -norm_w[k])
        cov = set()
        sel_norm = []
        for k in order_norm:
            tl = set(tuple(t) for t in cands[k]["tiles"])
            if tl.isdisjoint(cov):
                cov |= tl
                sel_norm.append(cands[k])
        tot["greedy_norm"] += count_tp([c["bbox"] for c in sel_norm])

        if i % 25 == 0:
            P(f"    ... recompute {i}/{len(grid_cache)} ({time.time()-t0:.0f}s)")
    return tot
"""

new_lines = lines[:start] + new_func.splitlines() + [""] + lines[end:]
target.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
print("SUCCESS: exp12_bridge_validation.py updated cleanly.")
