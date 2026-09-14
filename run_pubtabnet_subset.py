"""Download or reuse a PubTabNet subset and run measure_raw_consistency.run_on_pubtabnet_folder()

Saves images to `pubtab_samples/images/`, IOCR JSONs to `pubtab_samples/iocr_jsons/`,
creates a table_bboxes_by_image mapping, then calls the measurement function.
"""
import argparse
import json
import os
import sys
from itertools import islice
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser(description="Run raw consistency measurement on PubTabNet tables.")
parser.add_argument("--samples", type=int, default=300, help="Number of table images to process (default: 300)")
parser.add_argument("--device", type=str, default=None, help="Device to run inference on ('cuda' or 'cpu')")
args = parser.parse_args()

SAMPLES = args.samples
OUT_DIR = Path("pubtab_samples")
IMG_DIR = OUT_DIR / "images"
JSON_DIR = OUT_DIR / "iocr_jsons"
IMG_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

# 1. Discover already existing images
existing_pngs = sorted(list(IMG_DIR.glob("*.png")))
mapping = {}

print(f"Found {len(existing_pngs)} existing table images in {IMG_DIR}.")
for img_path in existing_pngs[:SAMPLES]:
    stem = img_path.stem
    json_path = JSON_DIR / f"{stem}_iocr.json"

    # Ensure JSON exists and has width/height
    w, h = Image.open(img_path).size
    if not json_path.exists():
        iocr = {"pages": [{"tokens": [], "width": w, "height": h}]}
        with open(json_path, "w") as fp:
            json.dump(iocr, fp)

    mapping[img_path.name] = [[0, 0, w, h]]

# 2. If more samples are requested than currently stored, stream from Hugging Face
needed = SAMPLES - len(mapping)
if needed > 0:
    print(f"Streaming {needed} additional samples from ajimeno/PubTabNet...")
    from datasets import load_dataset

    ds_stream = load_dataset("ajimeno/PubTabNet", split="train", streaming=True)
    added = 0
    for i, item in enumerate(ds_stream):
        if added >= needed:
            break

        filename = item.get("__key__") or item.get("filename") or f"pubtab_stream_{i}"
        stem = Path(str(filename)).stem
        img_name = f"{stem}.png"
        img_path = IMG_DIR / img_name

        if img_name in mapping:
            continue

        img = None
        for key in ("png", "img", "image", "image_file", "img_bytes"):
            if key in item:
                img = item[key]
                break
        if img is None:
            try:
                img = item["image"]
            except Exception:
                pass
        if img is None:
            continue

        try:
            img.save(img_path)
        except Exception:
            from io import BytesIO
            if isinstance(img, (bytes, bytearray)):
                im = Image.open(BytesIO(img))
                im.save(img_path)
            else:
                try:
                    import numpy as np
                    im = Image.fromarray(img)
                    im.save(img_path)
                except Exception:
                    continue

        w, h = Image.open(img_path).size
        iocr = {"pages": [{"tokens": [], "width": w, "height": h}]}
        json_path = JSON_DIR / f"{stem}_iocr.json"
        with open(json_path, "w") as fp:
            json.dump(iocr, fp)

        mapping[img_name] = [[0, 0, w, h]]
        added += 1

print(f"Prepared {len(mapping)} images for consistency evaluation.")

# 3. Import measurement library and model config
repo_root = os.path.join(os.path.dirname(__file__), "docling-ibm-models")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from tests.test_tf_predictor import test_config
from measure_raw_consistency import run_on_pubtabnet_folder

# 4. Download model artifacts if not cached
from huggingface_hub import snapshot_download
print("Ensuring model artifacts are available...")
download_path = snapshot_download(repo_id="ds4sd/docling-models", revision="v2.1.0")
save_dir = os.path.join(download_path, "model_artifacts/tableformer/fast")

# 5. Execute measurement on the folder
records = run_on_pubtabnet_folder(
    str(IMG_DIR),
    str(JSON_DIR),
    mapping,
    test_config,
    save_dir,
    device=args.device,
)
print("Complete. Summary and CSV report updated.")
