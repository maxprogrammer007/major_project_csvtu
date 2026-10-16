import sys
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

splerge_path = Path("d:/RRC/major_project_csvtu/AdvancedLiterateMachinery/DocumentUnderstanding/SPLERGE")
sys.path.insert(0, str(splerge_path))

from libs.split_model_hf import SplitModel
import libs.GTElement as gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SplitModel().to(device)
model.load_state_dict(torch.load(splerge_path / "model_out/split_model.pth", map_location=device)["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((960, 960)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_middle_of_groups(binary_array):
    result = np.zeros_like(binary_array)
    i = 0
    n = len(binary_array)
    while i < n:
        if binary_array[i] == 1:
            start = i
            while i < n and binary_array[i] == 1:
                i += 1
            result[(start + i - 1) // 2] = 1
        else:
            i += 1
    return result

def main():
    img_path = Path("d:/RRC/major_project_csvtu/pubtab_samples_tokens/images/pubtabnet_tok_554187.png")
    json_path = Path("d:/RRC/major_project_csvtu/pubtab_samples_tokens/iocr_jsons/pubtabnet_tok_554187_iocr.json")
    with open(json_path) as f:
        data = json.load(f)
    gt_tokens = data["pages"][0].get("tokens", [])
    gt_boxes = [tok["bbox"] for tok in gt_tokens]

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

    print(f"Image size: {orig_w} x {orig_h}")
    print(f"Row separators ({len(row_seps)}): {row_seps}")
    print(f"Col separators ({len(col_seps)}): {col_seps}")

    t_raw = gc.Table(0, 0, orig_w, orig_h)
    for r in row_seps: t_raw.gtRows.append(gc.Row(0, r, orig_w))
    for c in col_seps: t_raw.gtCols.append(gc.Column(c, 0, orig_h))
    t_raw.evaluateCells()

    num_rows = len(t_raw.gtCells)
    num_cols = len(t_raw.gtCells[0]) if num_rows > 0 else 0
    print(f"Elementary grid dimensions: {num_rows} rows x {num_cols} cols = {num_rows * num_cols} cells")

    # Let's inspect each row and the tokens falling near row boundaries
    # Notice that row boundaries divide the table into y-ranges:
    # row 0: y in [0, row_seps[0]]
    # row r: y in [row_seps[r-1], row_seps[r]]
    # row last: y in [row_seps[-1], orig_h]
    y_bounds = [0] + row_seps + [orig_h]
    x_bounds = [0] + col_seps + [orig_w]

    print("\nInspecting first 6 rows and tokens:")
    for r in range(min(8, num_rows)):
        y1, y2 = y_bounds[r], y_bounds[r+1]
        print(f"\n--- Grid Row {r}: y in [{y1}, {y2}] ---")
        tokens_in_row = []
        for tok_idx, tok in enumerate(gt_tokens):
            tb = tok["bbox"]
            # check vertical overlap
            ov = max(0, min(tb[3], y2) - max(tb[1], y1))
            if ov > 0:
                frac = ov / (tb[3] - tb[1])
                tokens_in_row.append((tok_idx, tok["text"], tb, frac))
        for tok_idx, text, tb, frac in tokens_in_row:
            print(f"   tok #{tok_idx} '{text}': y=[{tb[1]}, {tb[3]}], x=[{tb[0]}, {tb[2]}], frac_in_row={frac:.3f}")

if __name__ == "__main__":
    main()
