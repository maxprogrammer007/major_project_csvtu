import requests
import tarfile
from huggingface_hub import hf_hub_url
from pathlib import Path

repo_id = 'bsmock/pubtables-1m'
filename = 'PubTables-1M-Structure_Images_Test.tar.gz'
url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type='dataset')

out_dir = Path('pubtables_test_sample')
images_dir = out_dir / 'images'
images_dir.mkdir(parents=True, exist_ok=True)

target_count = 100
current_count = len(list(images_dir.glob('*.jpg')))
print(f"Currently have {current_count} images. Target: {target_count}")

if current_count < target_count:
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, stream=True, headers=headers)
    if response.status_code == 200:
        with tarfile.open(fileobj=response.raw, mode='r|gz') as tar:
            for member in tar:
                if member.name.endswith('.jpg'):
                    img_name = Path(member.name).name
                    img_path = images_dir / img_name
                    if not img_path.exists():
                        f = tar.extractfile(member)
                        with open(img_path, 'wb') as out_f:
                            out_f.write(f.read())
                    current_count = len(list(images_dir.glob('*.jpg')))
                    if current_count % 20 == 0:
                        print(f"Extracted {current_count}/{target_count} images...")
                    if current_count >= target_count:
                        break

print(f"Done! Total images: {len(list(images_dir.glob('*.jpg')))}")
