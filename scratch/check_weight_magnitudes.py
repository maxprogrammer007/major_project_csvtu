import json
from pathlib import Path
import numpy as np
import sys

BASE_DIR = Path("d:/RRC/major_project_csvtu")
sys.path.insert(0, str(BASE_DIR))

from run_full_metric_suite import build_candidates_and_weights, detect_ocr_tokens

with open(BASE_DIR / "scratch/splerge_100_tables_grid_cache.json", "r") as f:
    grid_cache = json.load(f)

all_weights = []
elem_weights = []
merge_weights = []
token_contribs = []

for stem, g_entry in list(grid_cache.items())[:25]:
    orig_w, orig_h = g_entry["orig_w"], g_entry["orig_h"]
    row_seps, col_seps = g_entry["row_seps"], g_entry["col_seps"]
    img_path = BASE_DIR / "pubtab_samples_tokens/images" / f"{stem}.png"
    ocr_tokens = detect_ocr_tokens(img_path)

    cands, weights, _, _, _, _ = build_candidates_and_weights(orig_w, orig_h, row_seps, col_seps, ocr_tokens)
    for c, w in zip(cands, weights):
        all_weights.append(w)
        if c["type"] == "elem":
            elem_weights.append(w)
        else:
            merge_weights.append(w)

all_w = np.array(all_weights)
elem_w = np.array(elem_weights)
merge_w = np.array(merge_weights)

print("=== SPLERGE Candidate Weight Distribution (25 tables sample) ===")
print(f"Total candidates: {len(all_w)} (Elementary: {len(elem_w)}, Merges: {len(merge_w)})")
print(f"Elementary weights: Mean={elem_w.mean():.4f}, Median={np.median(elem_w):.4f}, 75th={np.percentile(elem_w, 75):.4f}, Max={elem_w.max():.4f}")
print(f"Merge weights:      Mean={merge_w.mean():.4f}, Median={np.median(merge_w):.4f}, 75th={np.percentile(merge_w, 75):.4f}, Max={merge_w.max():.4f}")
print(f"Fraction of elementary weights > 0.1:  {(elem_w > 0.1).mean():.2%}")
print(f"Fraction of elementary weights > 1.0:  {(elem_w >= 1.0).mean():.2%}")
print(f"Fraction of merge weights > 1.0:       {(merge_w >= 1.0).mean():.2%}")
