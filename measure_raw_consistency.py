"""
H1 pilot measurement: how often are TableFormer's RAW predictions
structurally inconsistent, before MatchingPostProcessor repairs them?

This does NOT modify docling-ibm-models. It monkeypatches two functions
that TFPredictor.predict() already calls internally on every table:

  1. otsl_sqr_chk(rs_seq, logdebug)
       -> checks whether the raw predicted OTSL sequence is a valid
          rectangular grid (every row the same length). Currently called
          for debug logging only; its return value is discarded.

  2. TFPredictor._check_bbox_sync(prediction)
       -> checks whether the number of predicted bboxes matches the number
          of predicted cell tags. If not, _remove_bbox_span_desync() silently
          deletes "extra" boxes to force them to line up. This mismatch is
          a raw, quantifiable local-prediction inconsistency.

Both checks already run on every table. This script just records what they
find instead of letting the results disappear, and pairs them with a table
identifier so you get a per-table + aggregate consistency report.

--------------------------------------------------------------------------
TOKENS GATE — important constraint
--------------------------------------------------------------------------
TFPredictor.predict() contains this guard (around line 825-829 of tf_predictor.py):

    if len(prediction["bboxes"]) > 0:
        if len(iocr_page["tokens"]) > 0:   # <-- gate
            if self.enable_post_process:
                matching_details = self._post_processor.process(...)

Post-processing (overlap correction, orphan attachment, deduplication)
only fires if iocr_page["tokens"] is non-empty. When using the PubTabNet
Hugging Face mirror (ajimeno/PubTabNet), only PNG images are available —
no text tokens. So every generated IOCR JSON has tokens=[] and this gate
is NEVER crossed. This means:
  - otsl_square, bbox_sync, n_raw_overlaps, n_exact_duplicates are all
    measured on the raw cell_matcher output and are valid H1 signals.
  - MatchingPostProcessor.process() (orphan attachment, column deletion,
    etc.) is NOT exercised by these runs. To measure those, real OCR
    tokens are required (Tesseract/EasyOCR or a different dataset).

--------------------------------------------------------------------------
SETUP (run once)
--------------------------------------------------------------------------
    git clone https://github.com/docling-project/docling-ibm-models
    cd docling-ibm-models
    pip install -e ".[opencv-python-headless]"
    pip install huggingface_hub pytest
    # copy this script into the repo root, then:
    python measure_raw_consistency.py

The first run downloads model weights (~hundreds of MB) from Hugging Face
(ds4sd/docling-models), so it needs network access to huggingface.co.

--------------------------------------------------------------------------
WHAT THIS FIRST PASS COVERS
--------------------------------------------------------------------------
This smoke-test config uses the 3 sample tables already bundled in the repo
(tests/test_data/samples/) -- just enough to confirm the instrumentation
works end-to-end and see real, non-zero numbers. To scale up to a real
PubTabNet sample, see `run_on_pubtabnet_folder()` at the bottom: swap in a
folder of PubTabNet table crops + their IOCR-format JSON and table bboxes,
following the same iocr_page / table_bboxes shape used below.
"""

import csv
import json
import os
from pathlib import Path

import cv2
import sys
import copy

# --- 1. Import the library pieces we're going to monkeypatch -------------
import docling_ibm_models.tableformer.data_management.tf_predictor as tf_predictor_module
from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor

# --- 2b. Geometric overlap and duplicate-cell helpers ----------------------
# We compute overlap and duplicate counts directly on a copy of the raw
# table_cells returned by _cell_matcher.match_cells(), captured via the
# two-call strategy in predict_with_consistency_log(). We do NOT monkeypatch
# _find_overlapping — that would fire during the *second* predict() call
# (post-processing enabled) and overwrite the pre-repair count with a value
# taken mid-repair-pipeline. The two-call approach is strictly more precise.

_last_overlap = {}


def _boxes_overlap(box1, box2):
    """Reimplementation of the library's do_boxes_overlap() for local use."""
    B1, B2 = box1["bbox"], box2["bbox"]
    if B1[0] >= B2[2] or B1[2] <= B2[0] or B1[3] <= B2[1] or B1[1] >= B2[3]:
        return False
    return True


def _analyze_raw_cells(table_cells):
    """Single-pass analysis of raw predicted table_cells before any post-processing.

    Counts two disjoint categories for each pair (i, j) with i < j:
      - n_bbox_duplicates : pair has identical bbox lists. These are the artifacts
        that _deduplicate_cells() targets. MUST compare bboxes only — each cell has
        a unique cell_id, so whole-dict equality always returns False and would
        structurally guarantee 0 duplicates regardless of the true rate.
      - n_overlaps        : pair has distinct bboxes that geometrically overlap.
        Exact-bbox-equal pairs are excluded from this count by the elif.

    Returns
    -------
    (n_overlaps, n_bbox_duplicates) : both are pair counts, not cell counts.
    """
    n = len(table_cells)
    n_overlaps = 0
    n_bbox_dups = 0
    for i in range(n):
        b1 = table_cells[i].get("bbox")
        for j in range(i + 1, n):
            b2 = table_cells[j].get("bbox")
            if b1 == b2:               # identical predicted position → duplicate
                n_bbox_dups += 1
            elif _boxes_overlap(table_cells[i], table_cells[j]):
                n_overlaps += 1
    return n_overlaps, n_bbox_dups


# Keep thin wrappers for callers that still use the old names
def _count_raw_overlaps(table_cells):
    n_ov, _ = _analyze_raw_cells(table_cells)
    return n_ov


def _count_exact_duplicates(table_cells):
    _, n_dup = _analyze_raw_cells(table_cells)
    return n_dup

# --- 2. Install the instrumentation ---------------------------------------
# Two small per-call buffers, reset before each predict() call.
_last_square = {}
_last_sync = {}

_orig_otsl_sqr_chk = tf_predictor_module.otsl_sqr_chk


def _logged_otsl_sqr_chk(rs_list, logdebug):
    is_square = _orig_otsl_sqr_chk(rs_list, logdebug)
    _last_square["value"] = is_square
    # Count all OTSL cell tokens that correspond to table cells.
    # tf_predictor/_check_bbox_sync uses counts from tokens like fcel, ecel,
    # xcel, ched, rhed, srow when comparing to bbox count. Make our
    # diagnostic match that internal logic.
    ot_tokens = ("fcel", "ecel", "xcel", "ched", "rhed", "srow")
    _last_square["num_cells"] = sum(rs_list.count(t) for t in ot_tokens)
    return is_square


tf_predictor_module.otsl_sqr_chk = _logged_otsl_sqr_chk

_orig_check_bbox_sync = TFPredictor._check_bbox_sync


def _logged_check_bbox_sync(self, prediction):
    match, bboxes = _orig_check_bbox_sync(self, prediction)
    _last_sync["value"] = match
    _last_sync["num_bboxes_raw"] = len(prediction["bboxes"])
    return match, bboxes


TFPredictor._check_bbox_sync = _logged_check_bbox_sync


def predict_with_consistency_log(predictor, iocr_page, table_bbox, table_image, scale_factor, table_id):
    """Run predict() twice and capture pre-repair structural consistency signals.

    Strategy:
      1. First call with enable_post_process=False: captures the raw cell_matcher
         output (table_cells before any repair pipeline runs). Count overlaps and
         exact duplicates here — this is the cleanest pre-repair snapshot.
      2. Second call with enable_post_process=True: the real output returned to
         the caller. Post-processing only fires if iocr_page["tokens"] is non-empty
         (see tokens gate in tf_predictor.py:825). With tokens=[], this second call
         is equivalent to the first in terms of post-processing, but we keep both
         for correctness when real tokens are eventually provided.

    NOTE: We do NOT use the _find_overlapping monkeypatch to count overlaps.
    That would overwrite this carefully-captured pre-repair count with a value
    taken from mid-repair-pipeline during the second predict() call.
    """
    _last_square.clear()
    _last_sync.clear()
    _last_overlap.clear()

    _prev_post = getattr(predictor, "enable_post_process", True)

    # 1) Raw predict — no post-processing; captures clean pre-repair table_cells
    predictor.enable_post_process = False
    try:
        _, matching_details_raw = predictor.predict(
            iocr_page, table_bbox, table_image, scale_factor, None, False
        )
    finally:
        predictor.enable_post_process = _prev_post

    # Measure pre-repair geometric consistency (no deep-copy needed: we read bbox, don't mutate)
    try:
        raw_cells = matching_details_raw.get("table_cells", [])
        n_ov, n_dup = _analyze_raw_cells(raw_cells)
        _last_overlap["n_raw_overlaps"] = n_ov
        _last_overlap["n_raw_cells"] = len(raw_cells)
        _last_overlap["n_exact_duplicates"] = n_dup
    except Exception:
        _last_overlap.clear()

    # 2) Real predict — post-processing enabled (fires only if tokens non-empty)
    predictor.enable_post_process = True
    try:
        tf_output, matching_details = predictor.predict(
            iocr_page, table_bbox, table_image, scale_factor, None, True
        )
    finally:
        predictor.enable_post_process = _prev_post

    record = {
        "table_id": table_id,
        "otsl_square": _last_square.get("value"),
        "raw_cell_count": _last_square.get("num_cells"),
        "bbox_sync": _last_sync.get("value"),
        "raw_bbox_count": _last_sync.get("num_bboxes_raw"),
        # Pre-repair geometric signals (from the raw, no-post-process call):
        "n_raw_overlaps": _last_overlap.get("n_raw_overlaps"),
        "n_cells_at_overlap_check": _last_overlap.get("n_raw_cells"),
        # Duplicate-cell count: artifacts that _deduplicate_cells() fixes.
        # Invisible in n_raw_overlaps by design (exact dups excluded there).
        "n_exact_duplicates": _last_overlap.get("n_exact_duplicates"),
    }
    return tf_output, matching_details, record


# --- 3. Smoke test on the repo's bundled sample tables --------------------
def run_smoke_test():
    from huggingface_hub import snapshot_download

    # Reuse the exact working config from the repo's own tests/test_tf_predictor.py
    # (imported directly so we don't hand-copy the large wordmap dict).
    # Ensure the repo's tests package is importable when running from this
    # workspace root (the cloned repo lives in ./docling-ibm-models).
    repo_tests_path = os.path.join(os.path.dirname(__file__), "docling-ibm-models")
    if repo_tests_path not in sys.path:
        sys.path.insert(0, repo_tests_path)

    from tests.test_tf_predictor import test_config, docling_api_data

    # Normalize any relative paths in the test data to point into the
    # cloned repo directory so file reads succeed when running from this
    # workspace root.
    repo_root = os.path.join(os.path.dirname(__file__), "docling-ibm-models")

    def _to_repo_path(p):
        if p is None:
            return p
        if os.path.isabs(p):
            return p
        candidate = os.path.join(repo_root, p.lstrip("./\\"))
        return candidate if os.path.exists(candidate) else p

    # docling_api_data contains lists of relative paths; rewrite them in-place
    if "table_jsons" in docling_api_data:
        docling_api_data["table_jsons"] = [
            _to_repo_path(p) for p in docling_api_data["table_jsons"]
        ]
    if "png_images" in docling_api_data:
        docling_api_data["png_images"] = [
            _to_repo_path(p) for p in docling_api_data["png_images"]
        ]

    download_path = snapshot_download(repo_id="ds4sd/docling-models", revision="v2.1.0")
    save_dir = os.path.join(download_path, "model_artifacts/tableformer/fast")
    test_config["model"]["save_dir"] = save_dir

    predictor = TFPredictor(test_config, device="cpu", num_threads=2)

    records = []
    for table_json_fn, png_image_fn, table_bboxes in zip(
        docling_api_data["table_jsons"],
        docling_api_data["png_images"],
        docling_api_data["table_bboxes"],
    ):
        with open(table_json_fn, "r") as fp:
            iocr_page = json.load(fp)["pages"][0]
        iocr_page["image"] = cv2.imread(png_image_fn)

        page_image_resized, scale_factor = predictor.resize_img(iocr_page["image"], height=1024)

        for t, table_bbox in enumerate(table_bboxes):
            tb = [c * scale_factor for c in table_bbox]
            table_image = page_image_resized[round(tb[1]):round(tb[3]), round(tb[0]):round(tb[2])]
            table_id = f"{Path(png_image_fn).stem}_table{t}"

            _, _, record = predict_with_consistency_log(
                predictor, iocr_page, tb, table_image, scale_factor, table_id
            )
            records.append(record)
            print(record)

    write_report(records, "raw_consistency_report.csv")


def write_report(records, out_path):
    if not records:
        print("No records collected.")
        return
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    import statistics

    n = len(records)
    n_square = sum(1 for r in records if r.get("otsl_square") is True)
    n_sync = sum(1 for r in records if r.get("bbox_sync") is True)

    overlap_records = [r for r in records if r.get("n_raw_overlaps") is not None]
    n_with_overlaps = sum(1 for r in overlap_records if (r.get("n_raw_overlaps") or 0) > 0)

    print(f"\n=======================================================")
    print(f"--- Consistency Summary over {n} tables ---")
    print(f"OTSL grid valid (square):        {n_square}/{n} ({100*n_square/n:.1f}%)")
    print(f"bbox count matches cell count:   {n_sync}/{n} ({100*n_sync/n:.1f}%)")

    if overlap_records:
        n_ov = len(overlap_records)
        ov_vals = [(r.get("n_raw_overlaps") or 0) for r in overlap_records]
        total_overlaps = sum(ov_vals)
        mean_ov = total_overlaps / n_ov
        import statistics
        median_ov = statistics.median(ov_vals)

        # IQR-based outlier fence: Q3 + 3*(Q3-Q1). Avoids the circularity of
        # mean+3stdev when a single dominant outlier inflates both the mean and stdev.
        sorted_vals = sorted(ov_vals)
        q1 = statistics.median(sorted_vals[: n_ov // 2])
        q3 = statistics.median(sorted_vals[(n_ov + 1) // 2 :])
        iqr = q3 - q1
        # Use 3*IQR (wider than Tukey's 1.5) to avoid flagging moderate outliers
        iqr_fence = q3 + 3 * iqr if iqr > 0 else float("inf")

        outlier_records = [r for r in overlap_records if (r.get("n_raw_overlaps") or 0) > iqr_fence]
        normal_records  = [r for r in overlap_records if (r.get("n_raw_overlaps") or 0) <= iqr_fence]
        normal_vals = [(r.get("n_raw_overlaps") or 0) for r in normal_records]

        print(f"Tables with raw overlaps:        {n_with_overlaps}/{n_ov} ({100*n_with_overlaps/n_ov:.1f}%)")
        if normal_vals:
            mean_norm  = sum(normal_vals) / len(normal_vals)
            median_norm = statistics.median(normal_vals)
            print(f"Overlap count (ordinary):        mean={mean_norm:.1f}, median={median_norm:.1f}  [n={len(normal_vals)}]")
        print(f"  (full-sample mean={mean_ov:.1f}, median={median_ov:.1f}, total={total_overlaps})")

        # Catastrophic failures: kept visible as a separate named metric, not scrubbed.
        # A catastrophic prediction is one where nearly all predicted cell bboxes span
        # the entire table area (degenerate structural collapse, not minor geometric nudge).
        if outlier_records:
            print(f"Catastrophic failures (IQR fence={iqr_fence:.0f}): {len(outlier_records)}/{n_ov}")
            for r in outlier_records:
                print(f"    {r['table_id']}: overlaps={r.get('n_raw_overlaps')}, cells={r.get('raw_cell_count')}")
        else:
            print(f"Catastrophic failures (IQR fence={iqr_fence:.0f}): 0/{n_ov}")

        dup_records = [r for r in records if r.get("n_exact_duplicates") is not None]
        if dup_records:
            n_with_dups = sum(1 for r in dup_records if (r.get("n_exact_duplicates") or 0) > 0)
            total_dups = sum((r.get("n_exact_duplicates") or 0) for r in dup_records)
            n_dr = len(dup_records)
            print(f"Tables with bbox-dup cells:      {n_with_dups}/{n_dr} ({100*n_with_dups/n_dr:.1f}%)")
            print(f"Total bbox-dup cell pairs:       {total_dups} (mean {total_dups/n_dr:.2f} per table)")

    print(f"Report written to: {out_path}")
    print(f"=======================================================\n")



# --- 4. Extension point: run on a real PubTabNet sample --------------------
def run_on_pubtabnet_folder(
    image_dir, iocr_json_dir, table_bboxes_by_image, config, save_dir,
    device=None, out_csv=None,
):
    """
    image_dir            : folder of table-crop PNGs
    iocr_json_dir         : folder of matching IOCR-format JSONs
    table_bboxes_by_image : {image_filename: [[x1,y1,x2,y2], ...]}
    device               : 'cuda' or 'cpu' (auto-detected if None)
    out_csv              : output CSV filename (default: pubtabnet_raw_consistency_report.csv)
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if out_csv is None:
        out_csv = "pubtabnet_raw_consistency_report.csv"

    config["model"]["save_dir"] = save_dir
    print(f"Initializing TFPredictor on device: {device}...")
    predictor = TFPredictor(config, device=device, num_threads=4)

    records = []
    total_imgs = len(table_bboxes_by_image)

    for idx, (img_name, bboxes) in enumerate(table_bboxes_by_image.items(), start=1):
        png_path = os.path.join(image_dir, img_name)
        json_path = os.path.join(iocr_json_dir, Path(img_name).stem + "_iocr.json")

        with open(json_path, "r") as fp:
            iocr_page = json.load(fp)["pages"][0]
        iocr_page["image"] = cv2.imread(png_path)

        page_image_resized, scale_factor = predictor.resize_img(iocr_page["image"], height=1024)

        for t, table_bbox in enumerate(bboxes):
            tb = [c * scale_factor for c in table_bbox]
            table_image = page_image_resized[round(tb[1]):round(tb[3]), round(tb[0]):round(tb[2])]
            table_id = f"{Path(img_name).stem}_table{t}"
            _, _, record = predict_with_consistency_log(
                predictor, iocr_page, tb, table_image, scale_factor, table_id
            )
            records.append(record)
            print(
                f"[{idx}/{total_imgs}] {table_id}: "
                f"cells={record.get('raw_cell_count')}, "
                f"square={record.get('otsl_square')}, "
                f"sync={record.get('bbox_sync')}, "
                f"overlaps={record.get('n_raw_overlaps')}, "
                f"dups={record.get('n_exact_duplicates')}"
            )

        # Incrementally flush report every 10 images
        if idx % 10 == 0:
            write_report(records, out_csv)

    write_report(records, out_csv)
    return records


if __name__ == "__main__":
    run_smoke_test()
