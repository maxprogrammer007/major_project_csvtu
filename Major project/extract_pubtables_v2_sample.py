"""
Extract a paired 50/100-table test sample from PubTables-v2 (Smock et al., Dec 2025, arXiv:2512.10888).
Source repository: kensho/PubTables-v2
Files:
  - PubTables-v2_Cropped-Tables_test_xml-annotations.tar.gz (4.5 MB)
  - PubTables-v2_Cropped-Tables_test_images.tar.gz (Streaming)

Saves exclusively into: Major project/pubtables_v2_test_sample/
  - images/*.jpg
  - annotations/*.xml
"""

import os
import sys
import tarfile
import io
from pathlib import Path
import requests
from huggingface_hub import hf_hub_url

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "pubtables_v2_test_sample"
IMAGES_DIR = OUT_DIR / "images"
ANN_DIR = OUT_DIR / "annotations"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
ANN_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COUNT = 50

def extract_pubtables_v2_sample(target_count=TARGET_COUNT):
    current_images = len(list(IMAGES_DIR.glob("*.jpg")))
    current_anns = len(list(ANN_DIR.glob("*.xml")))
    print(f"Current PubTables-v2 cache: {current_images} images, {current_anns} annotations. Target: {target_count}")
    
    if current_images >= target_count and current_anns >= target_count:
        print("Target count already satisfied!")
        return
        
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # 1. Download and extract annotations tar (only 4.5 MB)
    print("1. Fetching test XML annotations from kensho/PubTables-v2 ...")
    xml_url = hf_hub_url("kensho/PubTables-v2", "PubTables-v2_Cropped-Tables_test_xml-annotations.tar.gz", repo_type="dataset")
    res_xml = requests.get(xml_url, headers=headers)
    if res_xml.status_code != 200:
        raise RuntimeError(f"Failed to fetch XML annotations tar: status {res_xml.status_code}")
        
    xml_stems = []
    xml_tar = tarfile.open(fileobj=io.BytesIO(res_xml.content), mode="r:gz")
    for m in xml_tar.getmembers():
        if m.name.endswith(".xml"):
            xml_stem = Path(m.name).stem
            xml_stems.append(xml_stem)
            
    # Target first N stems
    target_stems = set(xml_stems[:target_count])
    for m in xml_tar.getmembers():
        if m.name.endswith(".xml"):
            stem = Path(m.name).stem
            if stem in target_stems:
                out_xml = ANN_DIR / f"{stem}.xml"
                if not out_xml.exists():
                    f = xml_tar.extractfile(m)
                    with open(out_xml, "wb") as out_f:
                        out_f.write(f.read())
                        
    print(f"Extracted {len(list(ANN_DIR.glob('*.xml')))} annotations.")
    
    # 2. Stream matching images from images tar
    print("2. Streaming matching test images from kensho/PubTables-v2 ...")
    img_url = hf_hub_url("kensho/PubTables-v2", "PubTables-v2_Cropped-Tables_test_images.tar.gz", repo_type="dataset")
    res_img = requests.get(img_url, stream=True, headers=headers)
    if res_img.status_code != 200:
        raise RuntimeError(f"Failed to fetch images stream: status {res_img.status_code}")
        
    extracted_images = 0
    with tarfile.open(fileobj=res_img.raw, mode="r|gz") as tar:
        for m in tar:
            if m.name.endswith(".jpg"):
                stem = Path(m.name).stem
                if stem in target_stems:
                    out_jpg = IMAGES_DIR / f"{stem}.jpg"
                    if not out_jpg.exists():
                        f = tar.extractfile(m)
                        with open(out_jpg, "wb") as out_f:
                            out_f.write(f.read())
                    extracted_images += 1
                    if extracted_images % 10 == 0 or extracted_images == target_count:
                        print(f"Extracted {extracted_images}/{target_count} images...")
                    if extracted_images >= target_count:
                        break
                        
    print(f"Done! Final sample in {OUT_DIR}: {len(list(IMAGES_DIR.glob('*.jpg')))} images, {len(list(ANN_DIR.glob('*.xml')))} annotations.")

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else TARGET_COUNT
    extract_pubtables_v2_sample(count)
