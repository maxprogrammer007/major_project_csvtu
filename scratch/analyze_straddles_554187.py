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
model.load_state_dict(torch.load(splerge_path / "model_out/split_model.pth", map_location=device, weights_only=False)["model_state_dict"])
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

def box_intersection_area(b1, b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)

def main():
    img_path = Path("d:/RRC/major_project_csvtu/pubtab_samples_tokens/images/pubtabnet_tok_554187.png")
    json_path = Path("d:/RRC/major_project_csvtu/pubtab_samples_tokens/iocr_jsons/pubtabnet_tok_554187_iocr.json")
    with open(json_path) as f:
        data = json.load(f)
    gt_tokens = data["pages"][0].get("tokens", [])

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

    y_bounds = [0] + row_seps + [orig_h]
    x_bounds = [0] + col_seps + [orig_w]
    num_rows = len(row_seps) + 1
    num_cols = len(col_seps) + 1

    print(f"Table dimensions: {num_rows} rows x {num_cols} cols")
    print(f"Total tokens: {len(gt_tokens)}")

    # Check for each token: which elementary cell contains what fraction of it?
    straddling_tokens = []
    for tok_idx, tok in enumerate(gt_tokens):
        tb = tok["bbox"]
        t_area = (tb[2] - tb[0]) * (tb[3] - tb[1])
        cell_fractions = []
        for r in range(num_rows):
            for c in range(num_cols):
                cb = [x_bounds[c], y_bounds[r], x_bounds[c+1], y_bounds[r+1]]
                inter = box_intersection_area(tb, cb)
                if inter > 0:
                    cell_fractions.append(((r, c), inter / t_area, inter))
        
        cell_fractions.sort(key=lambda x: x[1], reverse=True)
        top_cell, top_frac, top_inter = cell_fractions[0]
        if top_frac < 0.80:
            straddling_tokens.append((tok_idx, tok["text"], tb, cell_fractions))

    print(f"\nTokens with max containment in ANY single elementary cell < 80%: {len(straddling_tokens)}")
    for item in straddling_tokens:
        tok_idx, text, tb, fracs = item
        print(f"Tok #{tok_idx} '{text}' bbox={tb}:")
        for cell, frac, inter in fracs:
            print(f"   Cell {cell} [x:{x_bounds[cell[1]]}-{x_bounds[cell[1]+1]}, y:{y_bounds[cell[0]]}-{y_bounds[cell[0]+1]}]: {frac*100:.1f}%")

    print("\nTokens with max containment in ANY single elementary cell >= 80%:")
    high_containment_count = len(gt_tokens) - len(straddling_tokens)
    print(f"{high_containment_count} / {len(gt_tokens)} ({high_containment_count/len(gt_tokens)*100:.1f}%) tokens are >= 80% contained in a SINGLE elementary cell!")

if __name__ == "__main__":
    main()
