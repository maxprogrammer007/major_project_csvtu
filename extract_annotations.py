import tarfile
from pathlib import Path

tar_path = Path(r'C:\Users\abhin\.cache\huggingface\hub\datasets--bsmock--pubtables-1m\snapshots\35b1c097807e0b07ec5313879b85956b7b3890db\PubTables-1M-Structure_Annotations_Test.tar.gz')
images_dir = Path('pubtables_test_sample/images')
ann_dir = Path('pubtables_test_sample/annotations')
ann_dir.mkdir(parents=True, exist_ok=True)

image_stems = {p.stem for p in images_dir.glob('*.jpg')}
print(f"Target annotations needed: {len(image_stems)}")

target_xmls = {f"{stem}.xml" for stem in image_stems}
extracted = 0

with tarfile.open(tar_path, 'r:gz') as tar:
    for member in tar:
        base_name = Path(member.name).name
        if base_name in target_xmls:
            out_file = ann_dir / base_name
            if not out_file.exists():
                f = tar.extractfile(member)
                with open(out_file, 'wb') as out_f:
                    out_f.write(f.read())
            extracted += 1
            if extracted >= len(target_xmls):
                break

print(f"Extracted {len(list(ann_dir.glob('*.xml')))} annotations matching images.")
