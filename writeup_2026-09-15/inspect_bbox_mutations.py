"""
Inspect all CellMatcher methods for bbox mutations and confirm the
full match_cells() chain does not touch table_cell['bbox'] coordinates
after they are built by _build_table_cells().
Also checks MatchingPostProcessor._run_intersection_match as it's the
second matching sub-path referenced in process().
"""
import inspect
import sys
sys.path.insert(0, 'docling-ibm-models')

from docling_ibm_models.tableformer.data_management.tf_cell_matcher import CellMatcher
from docling_ibm_models.tableformer.data_management.matching_post_processor import MatchingPostProcessor


def show_bbox_writes(cls, method_name):
    fn = getattr(cls, method_name, None)
    if fn is None:
        print(f"  [NOT FOUND] {cls.__name__}.{method_name}")
        return
    src = inspect.getsource(fn)
    lines = src.split('\n')
    # Lines that assign TO bbox (not just read it)
    # Patterns: ["bbox"] = , ["bbox"][X] =, cell["bbox"], table_cell["bbox"] =
    writes = []
    for i, l in enumerate(lines, 1):
        stripped = l.strip()
        if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'"):
            continue
        if 'bbox' in stripped and '=' in stripped:
            # Filter out comparisons (==, !=, >=, <=) and comprehensions, just assignments
            import re
            if re.search(r'bbox.*[^=!<>]=[^=]', stripped):
                writes.append((i, stripped))
    print(f"\n=== {cls.__name__}.{method_name} — bbox WRITE lines ===")
    if writes:
        for lineno, line in writes:
            print(f"  L{lineno:4d}: {line}")
    else:
        print("  (none — method does not write to any bbox field)")


# CellMatcher paths
for method in ['match_cells', 'match_cells_dummy', '_build_table_cells', '_translate_bboxes', '_intersection_over_pdf_match']:
    show_bbox_writes(CellMatcher, method)

# MatchingPostProcessor second sub-path
for method in ['_run_intersection_match', '_align_table_cells_to_pdf', '_move_cells_to_left_pos', '_pick_orphan_cells', '_attach_orphan_to_cell']:
    show_bbox_writes(MatchingPostProcessor, method)
