import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import sys

splerge_path = Path("d:/RRC/major_project_csvtu/AdvancedLiterateMachinery/DocumentUnderstanding/SPLERGE")
sys.path.insert(0, str(splerge_path))
from libs.split_model_hf import SplitModel
import libs.GTElement as gc

def get_middle_of_groups(binary_array):
    result = np.zeros_like(binary_array)
    i = 0
    n = len(binary_array)
    while i < n:
        if binary_array[i] == 1:
            start = i
            while i < n and binary_array[i] == 1: i += 1
            result[(start + i - 1) // 2] = 1
        else: i += 1
    return result

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SplitModel().to(device)
model.load_state_dict(torch.load(splerge_path / "model_out/split_model.pth", map_location=device)["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((960, 960)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_path = Path("d:/RRC/major_project_csvtu/pubtab_samples_tokens/images/pubtabnet_tok_554187.png")
json_path = Path("d:/RRC/major_project_csvtu/pubtab_samples_tokens/iocr_jsons/pubtabnet_tok_554187_iocr.json")

with open(json_path) as f:
    data = json.load(f)

tokens = data["pages"][0].get("tokens", [])
img = Image.open(img_path).convert("RGB")
orig_w, orig_h = img.size

with torch.no_grad():
    h_pred, v_pred = model(transform(img).unsqueeze(0).to(device))
    h_pred = h_pred.repeat_interleave(2, dim=1).squeeze(0).cpu().numpy()
    v_pred = v_pred.repeat_interleave(2, dim=1).squeeze(0).cpu().numpy()

h_splits = np.where(get_middle_of_groups((h_pred > 0.5).astype(float)) == 1)[0]
v_splits = np.where(get_middle_of_groups((v_pred > 0.5).astype(float)) == 1)[0]
row_seps = [int(y * orig_h / 960.0) for y in h_splits if 5 < int(y * orig_h / 960.0) < orig_h - 5]
col_seps = [int(x * orig_w / 960.0) for x in v_splits if 5 < int(x * orig_w / 960.0) < orig_w - 5]

print(f"Table 554187: Image size ({orig_w}, {orig_h})")
print(f"Predicted Row Cuts ({len(row_seps)}): {row_seps}")
print(f"Predicted Col Cuts ({len(col_seps)}): {col_seps}")

# Grid cells:
_t = gc.Table(0, 0, orig_w, orig_h)
for r in row_seps: _t.gtRows.append(gc.Row(0, r, orig_w))
for c in col_seps: _t.gtCols.append(gc.Column(c, 0, orig_h))
_t.evaluateCells()

print(f"Grid Rows: {len(_t.gtCells)}, Grid Cols: {len(_t.gtCells[0])}")
print("\nFirst 3 rows of grid cells:")
for r_idx in range(min(3, len(_t.gtCells))):
    row_str = [f"Col {c_idx}: [{c.x1},{c.y1},{c.x2},{c.y2}]" for c_idx, c in enumerate(_t.gtCells[r_idx])]
    print(f"  Row {r_idx}: " + " | ".join(row_str[:3]))

print("\nTokens in the upper section (y <= 80):")
for idx, tok in enumerate(tokens):
    b = tok["bbox"]
    if b[1] <= 80:
        print(f"  Tok #{idx}: text=\"{tok['text']}\", bbox={b}")
