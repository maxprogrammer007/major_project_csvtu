import sys
import os
import json
import time
from pathlib import Path

BASE_DIR = Path("d:/RRC/major_project_csvtu")
SPLERGE_PATH = BASE_DIR / "AdvancedLiterateMachinery/DocumentUnderstanding/SPLERGE"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SPLERGE_PATH))

import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
import cv2
from PIL import Image

import libs.GTElement as gc
import libs.merge_utility as mut
import merge as merge_mod

from teds_metric import TEDS
import grits

def detect_ocr_tokens(img_path: Path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 2))
    dilated = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    img_area = img.shape[0] * img.shape[1]
    tokens = []
    tok_id = 0
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        if area < 6 or area > img_area * 0.15:
            continue
        if w < 3 or h < 3:
            continue
        tokens.append({"id": tok_id, "text": "word", "bbox": [int(x), int(y), int(x + w), int(y + h)]})
        tok_id += 1
    return tokens

def box_intersection_area(b1, b2):
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2])
    y1 = min(b1[3], b2[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)

def build_candidates_and_weights(orig_w, orig_h, row_seps, col_seps, tokens):
    y_bounds = [0] + sorted(row_seps) + [orig_h]
    x_bounds = [0] + sorted(col_seps) + [orig_w]
    num_rows = len(y_bounds) - 1
    num_cols = len(x_bounds) - 1

    tile_to_idx = {}
    idx = 0
    for r in range(num_rows):
        for c in range(num_cols):
            tile_to_idx[(r, c)] = idx
            idx += 1
    num_tiles = idx

    candidates = []
    # 1. Elementary
    for r in range(num_rows):
        for c in range(num_cols):
            candidates.append({
                "type": "elem",
                "r1": r, "r2": r, "c1": c, "c2": c,
                "bbox": [x_bounds[c], y_bounds[r], x_bounds[c+1], y_bounds[r+1]],
                "tiles": [(r, c)]
            })

    # 2. Vertical spans (height = 2)
    for r in range(num_rows - 1):
        for c in range(num_cols):
            candidates.append({
                "type": "v_merge_2",
                "r1": r, "r2": r+1, "c1": c, "c2": c,
                "bbox": [x_bounds[c], y_bounds[r], x_bounds[c+1], y_bounds[r+2]],
                "tiles": [(r, c), (r+1, c)]
            })

    # 3. Horizontal spans (width = 2)
    for c in range(num_cols - 1):
        for r in range(num_rows):
            candidates.append({
                "type": "h_merge_2",
                "r1": r, "r2": r, "c1": c, "c2": c+1,
                "bbox": [x_bounds[c], y_bounds[r], x_bounds[c+2], y_bounds[r+1]],
                "tiles": [(r, c), (r, c+1)]
            })

    token_bboxes = [tok["bbox"] for tok in tokens]
    token_areas  = [(b[2]-b[0]) * (b[3]-b[1]) for b in token_bboxes]

    for cand in candidates:
        cb = cand["bbox"]
        contained = []
        for t_idx, (tb, t_area) in enumerate(zip(token_bboxes, token_areas)):
            if t_area <= 0: continue
            inter = box_intersection_area(tb, cb)
            if inter > 0:
                contained.append((t_idx, inter / t_area))
        cand["contained"] = contained

    elem_max_frac = {}
    for t_idx, (tb, t_area) in enumerate(zip(token_bboxes, token_areas)):
        if t_area <= 0:
            elem_max_frac[t_idx] = 0.0
            continue
        mf = 0.0
        for cand in candidates:
            if cand["type"] == "elem":
                mf = max(mf, box_intersection_area(tb, cand["bbox"]) / t_area)
        elem_max_frac[t_idx] = mf

    weights = np.zeros(len(candidates))
    for i, cand in enumerate(candidates):
        w = 0.0
        for t, f in cand["contained"]:
            if f >= 0.70: w += 1.0
            elif f >= 0.50: w += 0.5
        if cand["type"] != "elem":
            rescued = [t for t, f in cand["contained"]
                       if f >= 0.70 and elem_max_frac.get(t, 0.0) < 0.70]
            if len(rescued) > 0:
                w += 1.5 * len(rescued)
            else:
                w -= 0.05
        else:
            w += 0.01
        weights[i] = max(0.001, w)

    return candidates, weights, num_rows, num_cols, num_tiles, tile_to_idx

def solve_milp(candidates, weights, num_tiles, tile_to_idx):
    n = len(candidates)
    c = -weights
    row_ind, col_ind, data_vals = [], [], []
    for c_idx, cand in enumerate(candidates):
        for tile in cand["tiles"]:
            row_ind.append(tile_to_idx[tile])
            col_ind.append(c_idx)
            data_vals.append(1.0)
    A_sparse = sp.csc_matrix((data_vals, (row_ind, col_ind)), shape=(num_tiles, n))
    constraints = LinearConstraint(A_sparse, -np.inf, 1.0)
    integrality = np.ones(n, dtype=int)
    bounds = Bounds(0.0, 1.0)

    t0 = time.perf_counter()
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    solve_ms = (time.perf_counter() - t0) * 1000.0

    selected = [candidates[i] for i in np.where(res.x > 0.5)[0]]
    return selected, solve_ms

def solve_greedy(candidates, weights, seed=42):
    rng = np.random.RandomState(seed)
    n = len(candidates)
    jitter = rng.uniform(0.0, 1e-6, size=n)
    effective_keys = weights + jitter
    order = np.argsort(-effective_keys)

    selected = []
    covered = set()
    for idx in order:
        cand_tiles = set(candidates[idx]["tiles"])
        if cand_tiles.isdisjoint(covered):
            covered |= cand_tiles
            selected.append(candidates[idx])
    return selected

def canonical_grid_to_html(num_rows, num_cols, cells, text_map=None):
    grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    for cell in cells:
        r1, r2, c1, c2 = cell["r1"], cell["r2"], cell["c1"], cell["c2"]
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if r < num_rows and c < num_cols:
                    grid[r][c] = cell

    html_parts = ["<table>"]
    for r in range(num_rows):
        html_parts.append("<tr>")
        for c in range(num_cols):
            cell = grid[r][c]
            if cell is not None:
                if cell["r1"] == r and cell["c1"] == c:
                    r1, r2, c1, c2 = cell["r1"], cell["r2"], cell["c1"], cell["c2"]
                    rowspan = r2 - r1 + 1
                    colspan = c2 - c1 + 1
                    attrs = []
                    if rowspan > 1: attrs.append(f'rowspan="{rowspan}"')
                    if colspan > 1: attrs.append(f'colspan="{colspan}"')
                    attr_str = (" " + " ".join(attrs)) if attrs else ""
                    content = ""
                    if text_map and (r1, r2, c1, c2) in text_map:
                        content = text_map[(r1, r2, c1, c2)]
                    html_parts.append(f"<td{attr_str}>{content}</td>")
            else:
                html_parts.append("<td></td>")
        html_parts.append("</tr>")
    html_parts.append("</table>")
    return "".join(html_parts)

def assign_tokens_to_cells(cells, tokens):
    text_map = {}
    for cell in cells:
        cb = cell["bbox"]
        cell_toks = []
        for t in tokens:
            tb = t["bbox"]
            cx = (tb[0] + tb[2]) / 2.0
            cy = (tb[1] + tb[3]) / 2.0
            if cb[0] <= cx <= cb[2] and cb[1] <= cy <= cb[3]:
                cell_toks.append(t)
        # Sort reading order: y then x
        cell_toks.sort(key=lambda t: (t["bbox"][1], t["bbox"][0]))
        text = " ".join(t.get("text", "") for t in cell_toks if t.get("text"))
        text_map[(cell["r1"], cell["r2"], cell["c1"], cell["c2"])] = text
    return text_map

# Load caches
grid_cache = json.load(open(BASE_DIR / "scratch/splerge_100_tables_grid_cache.json"))
gt_html_cache = json.load(open(BASE_DIR / "scratch/pubtabnet_100_tables_gt_html.json"))

stem = list(grid_cache.keys())[0]
g_entry = grid_cache[stem]
gt_entry = gt_html_cache[stem]["html_data"]

gt_struct_tokens = gt_entry["structure"]["tokens"]
gt_struct_html = "<table>" + "".join(gt_struct_tokens) + "</table>"

teds_s = TEDS(structure_only=True, ignore_nodes=["thead", "tbody"])

orig_w, orig_h = g_entry["orig_w"], g_entry["orig_h"]
row_seps, col_seps = g_entry["row_seps"], g_entry["col_seps"]
gt_tokens = g_entry["gt_tokens"]

cand_gt, weights_gt, n_rows, n_cols, n_tiles, tile_to_idx = \
    build_candidates_and_weights(orig_w, orig_h, row_seps, col_seps, gt_tokens)

# Raw cells
raw_cells = [c for c in cand_gt if c["type"] == "elem"]
raw_html = canonical_grid_to_html(n_rows, n_cols, raw_cells)
teds_s_raw = teds_s.evaluate(raw_html, gt_struct_html)
grits_raw = grits.grits_from_html(gt_struct_html, raw_html)

# H2 GT ILP
h2gt_selected, solve_ms = solve_milp(cand_gt, weights_gt, n_tiles, tile_to_idx)
h2gt_html = canonical_grid_to_html(n_rows, n_cols, h2gt_selected)
teds_s_h2gt = teds_s.evaluate(h2gt_html, gt_struct_html)
grits_h2gt = grits.grits_from_html(gt_struct_html, h2gt_html)

# H2 Greedy
greedy_selected = solve_greedy(cand_gt, weights_gt, seed=42)
greedy_html = canonical_grid_to_html(n_rows, n_cols, greedy_selected)
teds_s_greedy = teds_s.evaluate(greedy_html, gt_struct_html)
grits_greedy = grits.grits_from_html(gt_struct_html, greedy_html)

# H1 Heuristic
raw_ocr = [[tok["id"], tok.get("text", ""), tok["bbox"][0], tok["bbox"][1], tok["bbox"][2], tok["bbox"][3]] for tok in gt_tokens]
ocr_clean = mut.clean_ocr_data(raw_ocr)
ocr_sentences = mut.make_sentences(ocr_clean)

t_merge = gc.Table(0, 0, orig_w, orig_h)
for r in row_seps: t_merge.gtRows.append(gc.Row(0, r, orig_w))
for c in col_seps: t_merge.gtCols.append(gc.Column(c, 0, orig_h))
t_merge.evaluateCells()
merge_mod.col_merging(t_merge, ocr_sentences)
merge_mod.row_merging(t_merge, ocr_sentences)
merge_mod.remove_redundant_seperators(t_merge, ocr_sentences, 0.2)
merge_mod.remove_redundant_seperators(t_merge, ocr_sentences, 0.4)
t_merge.merge_header_v2(ocr_sentences)
t_merge.evaluateCells()

h1_cells = [{"r1": c.startRow, "r2": c.endRow, "c1": c.startCol, "c2": c.endCol, "bbox": [c.x1, c.y1, c.x2, c.y2]}
            for row in t_merge.gtCells for c in row if not c.dontCare]
h1_html = canonical_grid_to_html(len(t_merge.gtCells), len(t_merge.gtCells[0]), h1_cells)
teds_s_h1 = teds_s.evaluate(h1_html, gt_struct_html)
grits_h1 = grits.grits_from_html(gt_struct_html, h1_html)

print(f"Table: {stem}")
print(f"Grid: {n_rows} rows x {n_cols} cols")
print(f"Raw:      TEDS-S = {teds_s_raw:.4f}, GriTS-Top = {grits_raw['grits_top']:.4f}")
print(f"H1-Heur:  TEDS-S = {teds_s_h1:.4f}, GriTS-Top = {grits_h1['grits_top']:.4f}")
print(f"H2-ILP:   TEDS-S = {teds_s_h2gt:.4f}, GriTS-Top = {grits_h2gt['grits_top']:.4f} (solve: {solve_ms:.1f}ms)")
print(f"H2-Greedy:TEDS-S = {teds_s_greedy:.4f}, GriTS-Top = {grits_greedy['grits_top']:.4f}")

