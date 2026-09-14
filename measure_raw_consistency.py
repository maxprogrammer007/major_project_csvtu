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


def _count_raw_overlaps(table_cells):
    """Count pairs of distinct (non-identical) cells whose bboxes overlap.
    Exact duplicate cells are excluded by construction — they are counted
    separately by _count_exact_duplicates().
    """
    cells_copy = copy.deepcopy(table_cells)
    n_overlaps = 0
    for i in range(len(cells_copy)):
        for j in range(i + 1, len(cells_copy)):
            if cells_copy[i] != cells_copy[j] and _boxes_overlap(cells_copy[i], cells_copy[j]):
                n_overlaps += 1
    return n_overlaps


def _count_exact_duplicates(table_cells):
    """Count how many cells are exact duplicates of another cell in the list.
    These are the artifacts that _deduplicate_cells() is designed to remove.
    A cell is counted as a duplicate if its bbox is identical to any earlier
    cell's bbox. Returns the total number of duplicate instances.
    """
    seen = []
    n_dups = 0
    for cell in table_cells:
        bbox = tuple(cell.get("bbox", []))
        if bbox in seen:
            n_dups += 1
        else:
            seen.append(bbox)
    return n_dups

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

    # Measure pre-repair geometric consistency on a deep copy (cells are mutable dicts)
    try:
        raw_cells = matching_details_raw.get("table_cells", [])
        raw_cells_copy = copy.deepcopy(raw_cells)
        _last_overlap["n_raw_overlaps"] = _count_raw_overlaps(raw_cells_copy)
        _last_overlap["n_raw_cells"] = len(raw_cells)
        _last_overlap["n_exact_duplicates"] = _count_exact_duplicates(raw_cells)
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
        median_ov = statistics.median(ov_vals)

        print(f"Tables with raw overlaps:        {n_with_overlaps}/{n_ov} ({100*n_with_overlaps/n_ov:.1f}%)")
        print(f"Overlap count — mean: {mean_ov:.1f},  median: {median_ov:.1f},  total: {total_overlaps}")

        # Flag outliers: any table with overlap_count > mean + 3*stdev (or > 500 hard cap)
        if n_ov > 1:
            try:
                stdev_ov = statistics.stdev(ov_vals)
                threshold = mean_ov + 3 * stdev_ov
            except Exception:
                threshold = max(ov_vals)
            outliers = [
                (r["table_id"], r.get("n_raw_overlaps"), r.get("raw_cell_count"))
                for r in overlap_records
                if (r.get("n_raw_overlaps") or 0) > threshold
            ]
            if outliers:
                print(f"  *** OUTLIERS (>mean+3stdev={threshold:.0f}):")
                for tid, ov, nc in outliers:
                    print(f"      {tid}: overlaps={ov}, cells={nc}")
                ov_vals_no_outlier = [v for v in ov_vals if v <= threshold]
                if ov_vals_no_outlier:
                    mean_clean = sum(ov_vals_no_outlier) / len(ov_vals_no_outlier)
                    median_clean = statistics.median(ov_vals_no_outlier)
                    print(f"  Excluding outliers — mean: {mean_clean:.1f}, median: {median_clean:.1f}")

        dup_records = [r for r in records if r.get("n_exact_duplicates") is not None]
        if dup_records:
            n_with_dups = sum(1 for r in dup_records if (r.get("n_exact_duplicates") or 0) > 0)
            total_dups = sum((r.get("n_exact_duplicates") or 0) for r in dup_records)
            n_dr = len(dup_records)
            print(f"Tables with exact dup cells:     {n_with_dups}/{n_dr} ({100*n_with_dups/n_dr:.1f}%)")
            print(f"Total exact dup cells:           {total_dups} (mean {total_dups/n_dr:.2f} per table)")

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
