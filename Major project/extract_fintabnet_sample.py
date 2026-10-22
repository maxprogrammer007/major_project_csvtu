"""
Extract a 100-table evaluation sample of FinTabNet.c from HuggingFace Hub.
Dataset: bsmock/FinTabNet.c (Brandon Smock et al., 2023)
Files: FinTabNet.c-Structure.tar.gz

Saves exclusively into: Major project/fintabnet_test_sample/
  - images/*.jpg
  - annotations/*.xml
"""

import os
import sys
import tarfile
from pathlib import Path
import requests
from huggingface_hub import hf_hub_url

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "fintabnet_test_sample"
IMAGES_DIR = OUT_DIR / "images"
ANN_DIR = OUT_DIR / "annotations"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
ANN_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COUNT = 100

def extract_fintabnet_sample(target_count=TARGET_COUNT):
    current_images = len(list(IMAGES_DIR.glob("*.jpg")))
    current_anns = len(list(ANN_DIR.glob("*.xml")))
    print(f"Current count: {current_images} images, {current_anns} annotations. Target: {target_count}")
    
    if current_images >= target_count and current_anns >= target_count:
        print("FinTabNet.c sample already extracted!")
        return
        
    url = hf_hub_url("bsmock/FinTabNet.c", "FinTabNet.c-Structure.tar.gz", repo_type="dataset")
    print(f"Connecting to HuggingFace stream: {url} ...")
    
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, stream=True, headers=headers)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch tar stream, status: {response.status_code}")
        
    print("Streaming and extracting matched images and annotations...")
    extracted_stems = set(p.stem for p in IMAGES_DIR.glob("*.jpg"))
    candidate_members = {}
    
    with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
        for member in tar:
            name = Path(member.name).name
            stem = Path(member.name).stem
            
            # Target validation set tables
            if "val" in member.name:
                if name.endswith(".xml") or name.endswith(".jpg"):
                    if stem not in candidate_members:
                        candidate_members[stem] = {}
                    if name.endswith(".xml"):
                        candidate_members[stem]['xml'] = member
                    elif name.endswith(".jpg"):
                        candidate_members[stem]['jpg'] = member
                        
                    # When both xml and jpg are found for a table
                    if 'xml' in candidate_members[stem] and 'jpg' in candidate_members[stem]:
                        xml_mem = candidate_members[stem]['xml']
                        jpg_mem = candidate_members[stem]['jpg']
                        
                        xml_path = ANN_DIR / f"{stem}.xml"
                        jpg_path = IMAGES_DIR / f"{stem}.jpg"
                        
                        if not xml_path.exists():
                            f = tar.extractfile(xml_mem)
                            with open(xml_path, "wb") as out_f:
                                out_f.write(f.read())
                                
                        if not jpg_path.exists():
                            f = tar.extractfile(jpg_mem)
                            with open(jpg_path, "wb") as out_f:
                                out_f.write(f.read())
                                
                        extracted_stems.add(stem)
                        del candidate_members[stem]
                        
                        if len(extracted_stems) % 10 == 0 or len(extracted_stems) == target_count:
                            print(f"Extracted {len(extracted_stems)}/{target_count} paired tables...")
                            
                        if len(extracted_stems) >= target_count:
                            break
                            
    print(f"Finished extracting FinTabNet.c test sample: {len(list(IMAGES_DIR.glob('*.jpg')))} images, {len(list(ANN_DIR.glob('*.xml')))} annotations.")

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else TARGET_COUNT
    extract_fintabnet_sample(count)
