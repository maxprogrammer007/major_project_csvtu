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

# --- 1. Import the library pieces we're going to monkeypatch -------------
import docling_ibm_models.tableformer.data_management.tf_predictor as tf_predictor_module
from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor

# --- 2. Install the instrumentation ---------------------------------------
# Two small per-call buffers, reset before each predict() call.
_last_square = {}
_last_sync = {}

_orig_otsl_sqr_chk = tf_predictor_module.otsl_sqr_chk


def _logged_otsl_sqr_chk(rs_list, logdebug):
    is_square = _orig_otsl_sqr_chk(rs_list, logdebug)
    _last_square["value"] = is_square
    _last_square["num_cells"] = rs_list.count("fcel") + rs_list.count("ecel")
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
    """Calls predictor.predict() normally and returns the usual outputs plus
    a record of the two pre-repair consistency signals for this one table."""
    _last_square.clear()
    _last_sync.clear()

    tf_output, matching_details = predictor.predict(iocr_page, table_bbox, table_image, scale_factor)

    record = {
        "table_id": table_id,
        "otsl_square": _last_square.get("value"),
        "raw_cell_count": _last_square.get("num_cells"),
        "bbox_sync": _last_sync.get("value"),
        "raw_bbox_count": _last_sync.get("num_bboxes_raw"),
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

    n = len(records)
    n_square = sum(1 for r in records if r["otsl_square"] is True)
    n_sync = sum(1 for r in records if r["bbox_sync"] is True)
    print(f"\n--- Summary over {n} tables ---")
    print(f"OTSL grid valid (square):     {n_square}/{n}  ({100*n_square/n:.1f}%)")
    print(f"bbox count matches cell count: {n_sync}/{n}  ({100*n_sync/n:.1f}%)")
    print(f"Report written to {out_path}")


# --- 4. Extension point: run on a real PubTabNet sample --------------------
def run_on_pubtabnet_folder(image_dir, iocr_json_dir, table_bboxes_by_image, config, save_dir):
    """
    image_dir            : folder of table-crop PNGs
    iocr_json_dir         : folder of matching IOCR-format JSONs (same shape as
                             docling_api_data["table_jsons"] above -- if PubTabNet's
                             own format differs, this is the piece to adapt)
    table_bboxes_by_image : {image_filename: [[x1,y1,x2,y2], ...]}
    """
    config["model"]["save_dir"] = save_dir
    predictor = TFPredictor(config, device="cpu", num_threads=4)

    records = []
    for img_name, bboxes in table_bboxes_by_image.items():
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

    write_report(records, "pubtabnet_raw_consistency_report.csv")
    return records


if __name__ == "__main__":
    run_smoke_test()
