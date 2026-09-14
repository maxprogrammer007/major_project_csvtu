"""Download a small PubTabNet subset and run measure_raw_consistency.run_on_pubtabnet_folder()

Saves images to `pubtab_samples/images/`, IOCR JSONs to `pubtab_samples/iocr_jsons/`,
creates a table_bboxes_by_image mapping, then calls the measurement function.
"""
import os
import json
from pathlib import Path

from datasets import load_dataset
from itertools import islice

SAMPLES = 3
OUT_DIR = Path("pubtab_samples")
IMG_DIR = OUT_DIR / "images"
JSON_DIR = OUT_DIR / "iocr_jsons"
IMG_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading up to {SAMPLES} samples from ajimeno/PubTabNet...")
# Use streaming to avoid downloading the entire PubTabNet archive.
ds_stream = load_dataset("ajimeno/PubTabNet", split="train", streaming=True)

mapping = {}
for i, item in enumerate(islice(ds_stream, SAMPLES)):
    # Use dataset-provided key/URL for filename if available
    filename = item.get("__key__") or item.get("filename") or f"pubtab_{i}"
    stem = Path(str(filename)).stem
    img_name = f"{stem}.png"
    img_path = IMG_DIR / img_name

    # item may contain a PIL Image or bytes under different keys
    img = None
    for key in ("png", "img", "image", "image_file", "img_bytes"):
        if key in item:
            img = item[key]
            break
    if img is None:
        # try 'image' in features mapping
        try:
            img = item["image"]
        except Exception:
            pass
    if img is None:
        print(f"Warning: no image found for sample {i}, skipping")
        continue

    # Save image
    try:
        img.save(img_path)
    except Exception:
        # img might be bytes
        from PIL import Image
        from io import BytesIO

        if isinstance(img, (bytes, bytearray)):
            im = Image.open(BytesIO(img))
            im.save(img_path)
        else:
            # fallback: convert via numpy array
            try:
                import numpy as np
                im = Image.fromarray(img)
                im.save(img_path)
            except Exception:
                print(f"Failed to save image for sample {i}")
                continue

    # create minimal IOCR JSON: empty tokens and page dimensions
    from PIL import Image
    w, h = Image.open(img_path).size
    iocr = {"pages": [{"tokens": [], "width": w, "height": h}]}
    json_path = JSON_DIR / f"{stem}_iocr.json"
    with open(json_path, "w") as fp:
        json.dump(iocr, fp)

    # bbox covering whole image
    mapping[img_name] = [[0, 0, w, h]]

print(f"Saved {len(mapping)} images to {IMG_DIR}")

# Now run the measurement function
print("Invoking measure_raw_consistency.run_on_pubtabnet_folder()... this may download models.")
from measure_raw_consistency import run_on_pubtabnet_folder

# import test_config from the repo tests
import sys
repo_root = os.path.join(os.path.dirname(__file__), "docling-ibm-models")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from tests.test_tf_predictor import test_config

# download model artifacts
from huggingface_hub import snapshot_download
print("Downloading model artifacts (if needed)...")
download_path = snapshot_download(repo_id="ds4sd/docling-models", revision="v2.1.0")
save_dir = os.path.join(download_path, "model_artifacts/tableformer/fast")

# run
records = run_on_pubtabnet_folder(str(IMG_DIR), str(JSON_DIR), mapping, test_config, save_dir)
print("Done. Wrote pubtabnet_raw_consistency_report.csv")
