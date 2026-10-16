import json
from pathlib import Path
import sys

BASE_DIR = Path("d:/RRC/major_project_csvtu")
SPLERGE_PATH = BASE_DIR / "AdvancedLiterateMachinery/DocumentUnderstanding/SPLERGE"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SPLERGE_PATH))

from teds_metric import TEDS
from lxml import html
from run_full_metric_suite import (
    detect_ocr_tokens, build_candidates_and_weights, solve_milp,
    canonical_grid_to_html, run_splerge_heuristic, evaluate_matching
)

with open(BASE_DIR / "scratch/splerge_100_tables_grid_cache.json", "r") as f:
    grid_cache = json.load(f)
with open(BASE_DIR / "scratch/pubtabnet_100_tables_gt_html.json", "r") as f:
    gt_cache = json.load(f)

teds_eval = TEDS(structure_only=True)

for table_id in ["pubtabnet_tok_550868", "pubtabnet_tok_597425", "pubtabnet_tok_586722"]:
    grid_info = grid_cache[table_id]
    gt_info = gt_cache[table_id]

    orig_w = grid_info["orig_w"]
    orig_h = grid_info["orig_h"]
    row_seps = grid_info["row_seps"]
    col_seps = grid_info["col_seps"]
    n_rows = len(row_seps) + 1
    n_cols = len(col_seps) + 1

    img_path = BASE_DIR / "pubtab_samples_tokens/images" / f"{table_id}.png"
    ocr_tokens = detect_ocr_tokens(img_path)

    gt_boxes = grid_info["gt_boxes"]
    gt_entry = gt_info["html_data"]
    gt_struct_tokens = gt_entry["structure"]["tokens"]
    gt_struct_html = "<table>" + "".join(gt_struct_tokens) + "</table>"
    gt_root = html.fromstring(gt_struct_html)

    # H1
    h1_cells, h1_rows, h1_cols = run_splerge_heuristic(orig_w, orig_h, row_seps, col_seps, ocr_tokens)
    h1_html = canonical_grid_to_html(h1_rows, h1_cols, h1_cells)
    h1_boxes = [c["bbox"] for c in h1_cells]
    _, h1_tp, _, _, h1_prec, h1_rec, h1_f1 = evaluate_matching(h1_boxes, gt_boxes)
    h1_root = html.fromstring(h1_html)
    h1_teds = teds_eval.evaluate(h1_html, gt_struct_html)

    # H2
    cand_ocr, w_ocr, n_rows, n_cols, n_tiles, tile_idx = build_candidates_and_weights(orig_w, orig_h, row_seps, col_seps, ocr_tokens)
    h2_selected, _ = solve_milp(cand_ocr, w_ocr, n_tiles, tile_idx)
    h2_html = canonical_grid_to_html(n_rows, n_cols, h2_selected)
    h2_boxes = [c["bbox"] for c in h2_selected]
    _, h2_tp, _, _, h2_prec, h2_rec, h2_f1 = evaluate_matching(h2_boxes, gt_boxes)
    h2_root = html.fromstring(h2_html)
    h2_teds = teds_eval.evaluate(h2_html, gt_struct_html)

    print(f"\n=======================================================")
    print(f"Table ID: {table_id}")
    print(f"Ground Truth:      {len(gt_boxes):3d} cells | {len(gt_root.xpath('//tr')):2d} rows x {len(gt_root.xpath('//tr[1]/td')):2d} cols | {len(list(gt_root.iter())):3d} tree nodes")
    print(f"Split Cuts (Raw):  {len(gt_boxes):3d} cells | {n_rows:2d} rows x {n_cols:2d} cols | {n_rows*n_cols:3d} elementary tiles")
    print(f"H1 Heuristic:      {h1_tp:3d} TPs ({h1_rec:.1%}) | {h1_rows:2d} rows x {h1_cols:2d} cols | {len(list(h1_root.iter())):3d} tree nodes | TEDS-S = {h1_teds:.4f}")
    print(f"H2 Exact ILP:      {h2_tp:3d} TPs ({h2_rec:.1%}) | {n_rows:2d} rows x {n_cols:2d} cols | {len(list(h2_root.iter())):3d} tree nodes | TEDS-S = {h2_teds:.4f}")
    print(f"Delta (H2 - H1):   +{h2_tp - h1_tp:2d} cells recovered! | TEDS-S difference: {h2_teds - h1_teds:+.4f}")
