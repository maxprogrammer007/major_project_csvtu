import json
from pathlib import Path
import sys

BASE_DIR = Path("d:/RRC/major_project_csvtu")
SPLERGE_PATH = BASE_DIR / "AdvancedLiterateMachinery/DocumentUnderstanding/SPLERGE"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SPLERGE_PATH))

from run_full_metric_suite import (
    detect_ocr_tokens, build_candidates_and_weights, solve_milp,
    canonical_grid_to_html, run_splerge_heuristic, evaluate_matching
)

with open(BASE_DIR / "scratch/splerge_100_tables_grid_cache.json", "r") as f:
    grid_cache = json.load(f)
with open(BASE_DIR / "scratch/pubtabnet_100_tables_gt_html.json", "r") as f:
    gt_cache = json.load(f)

table_id = "pubtabnet_tok_597425"
grid_info = grid_cache[table_id]
gt_info = gt_cache[table_id]

orig_w = grid_info["orig_w"]
orig_h = grid_info["orig_h"]
row_seps = grid_info["row_seps"]
col_seps = grid_info["col_seps"]
gt_boxes = grid_info["gt_boxes"]

img_path = BASE_DIR / "pubtab_samples_tokens/images" / f"{table_id}.png"
ocr_tokens = detect_ocr_tokens(img_path)

print(f"=== Eyeballing {table_id} ===")
print(f"Image dimensions: {orig_w} x {orig_h}")
print(f"Ground truth cells: {len(gt_boxes)}")
print(f"Row cuts ({len(row_seps)}): {row_seps}")
print(f"Col cuts ({len(col_seps)}): {col_seps}")

# H1
h1_cells, h1_rows, h1_cols = run_splerge_heuristic(orig_w, orig_h, row_seps, col_seps, ocr_tokens)
h1_boxes = [c["bbox"] for c in h1_cells]
m_h1, h1_tp, _, _, h1_prec, h1_rec, h1_f1 = evaluate_matching(h1_boxes, gt_boxes)

print(f"\nH1 Heuristic output: {len(h1_cells)} cells, {h1_rows} rows x {h1_cols} cols")
print(f"H1 True Positives (@0.25): {h1_tp} (Recall: {h1_rec:.2%})")
print("H1 Cell bboxes:")
for i, c in enumerate(h1_cells[:10]):
    print(f"  Cell {i}: r=[{c['r1']},{c['r2']}], c=[{c['c1']},{c['c2']}], bbox={c['bbox']}")

# Raw
cand_ocr, w_ocr, n_rows, n_cols, n_tiles, tile_idx = build_candidates_and_weights(orig_w, orig_h, row_seps, col_seps, ocr_tokens)
raw_cells = [c for c in cand_ocr if c["type"] == "elem"]
raw_boxes = [c["bbox"] for c in raw_cells]
_, raw_tp, _, _, raw_prec, raw_rec, raw_f1 = evaluate_matching(raw_boxes, gt_boxes)
print(f"\nRaw Split Grid: {len(raw_cells)} cells, {n_rows} rows x {n_cols} cols")
print(f"Raw True Positives (@0.25): {raw_tp} (Recall: {raw_rec:.2%})")

# H2
h2_selected, _ = solve_milp(cand_ocr, w_ocr, n_tiles, tile_idx)
h2_boxes = [c["bbox"] for c in h2_selected]
_, h2_tp, _, _, h2_prec, h2_rec, h2_f1 = evaluate_matching(h2_boxes, gt_boxes)
print(f"\nH2 Exact ILP: {len(h2_selected)} cells, {n_rows} rows x {n_cols} cols")
print(f"H2 True Positives (@0.25): {h2_tp} (Recall: {h2_rec:.2%})")

print(f"\nExact Delta: Raw={raw_tp}, H1={h1_tp}, H2={h2_tp}")
print(f"Valid raw cells destroyed by H1: {raw_tp} - {h1_tp} = {raw_tp - h1_tp}")
print(f"Valid cells recovered by H2:    {h2_tp} - {h1_tp} = {h2_tp - h1_tp}")
