"""
Compare TableFormer raw and post-repair cell counts against PubTabNet ground truth
for the 5 catastrophic-collapse tables and a sample of ordinary tables.

Ground truth comes from apoidea/pubtabnet-html html.cells:
  - gt_total_cells   : all cells (including empty/span markers with no bbox)
  - gt_nonempty_cells: cells that have a bbox (these are what IOCR tokens were built from)

Comparison:
  raw_cell_count   : TableFormer OTSL prediction cell count (before repair)
  post_cell_count  : cells remaining after MatchingPostProcessor
  gt_total_cells   : ground truth total (rows x cols, includes structural empties)
  gt_nonempty_cells: ground truth non-empty (cells containing actual text/content)

This tells us whether 345->12 for _629558 means:
  A) 345 was wild over-prediction AND 12 is wild over-correction (both wrong)
  B) 345 was the pathology, 12 is closer to correct (deduplication worked)
"""
import ast, json, sys
from pathlib import Path

# The 5 catastrophic tables and a handful of ordinary tables for comparison
TARGET_IMGIDS = {
    # catastrophic
    629558, 634445, 670433, 677385, 729453,
    # ordinary: pick a few with varied raw counts from pre_post report
    639579,   # introduced a new overlap (raw=0, post=1)
    622581,   # raw=9, post=0 (ordinary, good recovery)
    624025,   # raw=18, post=0
    722429,   # raw=21, post=1 (residual overlap)
    629558,   # already catastrophic, listed again is fine (set dedupes)
}

# Load what we already measured
PRE_POST_CSV = Path('pubtabnet_pre_post_repair_report.csv')
import csv
measured = {}
with open(PRE_POST_CSV, newline='') as f:
    for r in csv.DictReader(f):
        stem = r['table_id'].replace('_table0', '')
        imgid = int(stem.replace('pubtabnet_tok_', ''))
        measured[imgid] = {
            'raw_cells':  int(r['raw_cell_count']),
            'post_cells': int(r['post_cell_count']),
            'raw_ov':     int(r['n_raw_overlaps']),
            'post_ov':    int(r['n_post_overlaps']),
            'is_catastrophic': r['is_catastrophic'] == 'True',
        }

# Stream from apoidea/pubtabnet-html to get ground truth
from datasets import load_dataset

def parse_html(html_str):
    try:
        return json.loads(html_str)
    except Exception:
        try:
            return ast.literal_eval(html_str)
        except Exception:
            return {}

print("Streaming apoidea/pubtabnet-html validation split to find target tables...")
ds = load_dataset('apoidea/pubtabnet-html', split='validation', streaming=True)

results = {}
found = set()

for item in ds:
    imgid = item.get('imgid')
    if imgid not in TARGET_IMGIDS:
        continue

    html_data = parse_html(item.get('html', '{}'))
    cells = html_data.get('cells', [])

    gt_total    = len(cells)
    gt_nonempty = sum(1 for c in cells if c.get('bbox') and len(c.get('bbox', [])) == 4)

    results[imgid] = {
        'gt_total_cells':    gt_total,
        'gt_nonempty_cells': gt_nonempty,
    }
    found.add(imgid)
    print(f"  Found imgid={imgid}: gt_total={gt_total}, gt_nonempty={gt_nonempty}")

    if found >= TARGET_IMGIDS:
        break

# Also check train split for any not found in validation
missing = TARGET_IMGIDS - found
if missing:
    print(f"\nNot found in validation, checking train split for: {missing}")
    ds_train = load_dataset('apoidea/pubtabnet-html', split='train', streaming=True)
    for item in ds_train:
        imgid = item.get('imgid')
        if imgid not in missing:
            continue
        html_data = parse_html(item.get('html', '{}'))
        cells = html_data.get('cells', [])
        gt_total    = len(cells)
        gt_nonempty = sum(1 for c in cells if c.get('bbox') and len(c.get('bbox', [])) == 4)
        results[imgid] = {'gt_total_cells': gt_total, 'gt_nonempty_cells': gt_nonempty}
        missing.discard(imgid)
        print(f"  Found imgid={imgid} (train): gt_total={gt_total}, gt_nonempty={gt_nonempty}")
        if not missing:
            break

# --- Print comparison table ---
print(f"\n{'='*90}")
print(f"Ground Truth vs TableFormer Raw vs Post-Repair Cell Counts")
print(f"{'='*90}")
print(f"{'imgid':>10} {'flag':<14} {'gt_total':>10} {'gt_nonempty':>12} {'raw_cells':>11} {'post_cells':>11} {'raw_ov':>8} {'post_ov':>8}")
print('-'*90)

CATASTROPHIC_IDS = {629558, 634445, 670433, 677385, 729453}

for imgid in sorted(results.keys()):
    gt = results[imgid]
    m  = measured.get(imgid, {})
    flag = '[CATASTROPHIC]' if imgid in CATASTROPHIC_IDS else '[ordinary]'
    print(f"{imgid:>10} {flag:<14} "
          f"{gt['gt_total_cells']:>10} {gt['gt_nonempty_cells']:>12} "
          f"{m.get('raw_cells','?'):>11} {m.get('post_cells','?'):>11} "
          f"{m.get('raw_ov','?'):>8} {m.get('post_ov','?'):>8}")

print()
print("Legend:")
print("  gt_total_cells   = all cells in ground truth (including empty/span structural cells)")
print("  gt_nonempty_cells= GT cells with actual bbox (content cells; used for IOCR tokens)")
print("  raw_cells        = TableFormer OTSL prediction count (before any repair)")
print("  post_cells       = cells remaining after MatchingPostProcessor")
