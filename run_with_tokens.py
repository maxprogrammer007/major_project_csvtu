"""Run the raw consistency measurement on PubTabNet with REAL per-cell tokens.

Source: apoidea/pubtabnet-html (Parquet-backed, validation + train splits)
  - Each item has: image (PIL), html (JSON string), html_table, split, imgid
  - html JSON contains: {"cells": [{"tokens": [...], "bbox": [x0,y0,x1,y1]}, ...]}
  - bbox is only present on non-empty cells
  - tokens are character-level (e.g. ['A', 'g', 'e', ...])

IOCR token format required by TFPredictor/CellMatcher:
  - iocr_page["tokens"]: list of dicts, each with "id", "bbox", "text"
  - bbox: [x0, y0, x1, y1] (list format accepted directly)
  - text: concatenated cell text

TOKENS GATE: With real tokens, MatchingPostProcessor.process() will actually run,
enabling measurement of orphan-attachment and deduplication stages.

Usage:
    python run_with_tokens.py                    # 100 samples (default)
    python run_with_tokens.py --samples 300      # 300 samples
    python run_with_tokens.py --inspect-schema   # print schema of 1 item and exit
    python run_with_tokens.py --device cpu       # force CPU
"""
import argparse
import ast
import json
import os
import sys
from pathlib import Path

import cv2
from PIL import Image

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="PubTabNet real-token consistency measurement.")
parser.add_argument("--samples", type=int, default=100,
                    help="Number of table images to process (default: 100)")
parser.add_argument("--device", type=str, default=None,
                    help="Inference device: 'cuda' or 'cpu' (auto-detected if unset)")
parser.add_argument("--inspect-schema", action="store_true",
                    help="Stream one item, print its full structure, then exit")
args = parser.parse_args()

OUT_DIR = Path("pubtab_samples_tokens")
IMG_DIR = OUT_DIR / "images"
JSON_DIR = OUT_DIR / "iocr_jsons"
IMG_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset source ────────────────────────────────────────────────────────────
DATASET_ID = "apoidea/pubtabnet-html"
DATASET_SPLIT = "validation"   # 894 tables, enough for robust measurement


def _parse_html_field(html_str):
    """Parse the html field — it's a JSON-encoded string in this mirror.
    Falls back to ast.literal_eval if json.loads fails (single-quoted repr).
    Returns a dict with key 'cells'.
    """
    try:
        return json.loads(html_str)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(html_str)
        except Exception:
            return {}


def _extract_iocr_tokens(html_str, img_w, img_h):
    """Convert PubTabNet html cell annotations into Docling IOCR token format.

    PubTabNet cells with no bbox are empty (header spacers etc.) — skip them.
    Cell text is character-level; we join into a single string per cell.
    Bbox coords are already pixel-absolute [x0,y0,x1,y1] relative to the
    table-crop image.

    Returns:
        tokens: list of {"id": int, "text": str, "bbox": [x0,y0,x1,y1]}
        table_bbox: [0, 0, img_w, img_h]  (whole-image bbox for this crop)
    """
    data = _parse_html_field(html_str)
    cells = data.get("cells", [])
    tokens = []
    tok_id = 0
    for cell in cells:
        bbox = cell.get("bbox")
        if not bbox or len(bbox) < 4:
            continue  # empty cell — no spatial anchor
        raw_tokens = cell.get("tokens", [])
        # Strip HTML tags (<b>, </b> etc.) from token characters
        text_chars = [t for t in raw_tokens if not (t.startswith("<") and t.endswith(">"))]
        text = "".join(text_chars).strip()
        if not text:
            text = " "  # keep a non-empty placeholder so the token is counted
        tokens.append({
            "id": tok_id,
            "text": text,
            "bbox": list(bbox),   # [x0, y0, x1, y1] — list format accepted by CellMatcher
        })
        tok_id += 1
    return tokens, [0, 0, img_w, img_h]


# ── Schema inspection ─────────────────────────────────────────────────────────
if args.inspect_schema:
    from datasets import load_dataset
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=True)
    print(f"\nFeatures: {ds.features}\n")
    item = next(iter(ds))
    print("Keys:", list(item.keys()))
    for k, v in item.items():
        t = type(v).__name__
        r = repr(v)
        print(f"\n  {k}: {t}")
        print(f"    {r[:500]}")
    # Show parsed html structure
    html_data = _parse_html_field(item.get("html", "{}"))
    cells = html_data.get("cells", [])
    print(f"\nParsed html: {len(cells)} cells")
    for i, c in enumerate(cells[:5]):
        print(f"  cell[{i}]: bbox={c.get('bbox')} tokens={c.get('tokens', [])[:10]}")
    print("\nSchema inspection done. Exiting.")
    sys.exit(0)

# ── Stream and prepare samples ────────────────────────────────────────────────
mapping = {}   # img_name -> [[x0,y0,x1,y1]]

# First check for already-cached samples
existing_pngs = sorted(list(IMG_DIR.glob("*.png")))
for img_path in existing_pngs:
    if len(mapping) >= args.samples:
        break
    json_path = JSON_DIR / f"{img_path.stem}_iocr.json"
    if json_path.exists():
        try:
            with open(json_path, "r") as fp:
                data = json.load(fp)["pages"][0]
                w, h = data["width"], data["height"]
                tokens = data.get("tokens", [])
                if tokens:
                    mapping[img_path.name] = [[0, 0, w, h]]
        except Exception:
            pass

if mapping:
    print(f"Discovered {len(mapping)} already-cached token sample pairs in {OUT_DIR}.")

needed = args.samples - len(mapping)
if needed > 0:
    from datasets import load_dataset
    print(f"Streaming {needed} additional samples from {DATASET_ID} ({DATASET_SPLIT})...")
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=True)
    prepared = 0

    for i, item in enumerate(ds):
        if len(mapping) >= args.samples:
            break

        img_id = item.get("imgid", i)
        img_name = f"pubtabnet_tok_{img_id}.png"
        img_path = IMG_DIR / img_name
        json_path = JSON_DIR / f"pubtabnet_tok_{img_id}_iocr.json"

        if img_name in mapping:
            continue

        # Save image
        pil_img = item.get("image")
        if pil_img is None:
            continue
        try:
            pil_img.save(img_path)
        except Exception as e:
            continue

        w, h = pil_img.size
        html_str = item.get("html", "{}")
        tokens, table_bbox = _extract_iocr_tokens(html_str, w, h)

        if not tokens:
            img_path.unlink(missing_ok=True)
            continue

        # Build IOCR JSON in the format TFPredictor expects
        iocr = {
            "pages": [{
                "tokens": tokens,
                "width": w,
                "height": h,
            }]
        }
        with open(json_path, "w") as fp:
            json.dump(iocr, fp)

        mapping[img_name] = [table_bbox]
        prepared += 1
        if prepared % 25 == 0:
            print(f"  Prepared {prepared}/{needed} new samples...")

print(f"\nTotal ready: {len(mapping)} images with real cell tokens.")
print(f"  Tokens gate will be OPEN (tokens > 0) for all these samples.")
print(f"  MatchingPostProcessor.process() WILL run — full repair pipeline exercised.\n")

if not mapping:
    print("No usable samples found. Exiting.")
    sys.exit(1)

# ── Model config ──────────────────────────────────────────────────────────────
repo_root = os.path.join(os.path.dirname(__file__), "docling-ibm-models")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from tests.test_tf_predictor import test_config
from measure_raw_consistency import run_on_pubtabnet_folder

from huggingface_hub import snapshot_download
print("Ensuring model artifacts are available...")
download_path = snapshot_download(repo_id="ds4sd/docling-models", revision="v2.1.0")
save_dir = os.path.join(download_path, "model_artifacts/tableformer/fast")

# ── Run measurement ───────────────────────────────────────────────────────────
records = run_on_pubtabnet_folder(
    str(IMG_DIR),
    str(JSON_DIR),
    mapping,
    test_config,
    save_dir,
    device=args.device,
    out_csv="pubtabnet_tokens_consistency_report.csv",
)
print("Complete. Results in pubtabnet_tokens_consistency_report.csv")
