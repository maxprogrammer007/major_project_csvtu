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

test_dir = Path("d:/RRC/major_project_csvtu/pubtab_samples_tokens/images")
json_dir = Path("d:/RRC/major_project_csvtu/pubtab_samples_tokens/iocr_jsons")
test_images = sorted(list(test_dir.glob("*.png")))[:15]

print("Comparing IoU vs Containment on first 15 tables:")
for img_path in test_images:
    with open(json_dir / (img_path.stem + "_iocr.json")) as f:
        data = json.load(f)
    gt_boxes = [tok["bbox"] for tok in data["pages"][0].get("tokens", [])]
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
    
    _t = gc.Table(0, 0, orig_w, orig_h)
    for r in row_seps: _t.gtRows.append(gc.Row(0, r, orig_w))
    for c in col_seps: _t.gtCols.append(gc.Column(c, 0, orig_h))
    _t.evaluateCells()
    raw_cells = [[c.x1, c.y1, c.x2, c.y2] for row in _t.gtCells for c in row if not c.dontCare]
    
    # Check containment of GT boxes inside raw cells
    contained_count = 0
    iou25_count = 0
    for gb in gt_boxes:
        gt_area = (gb[2] - gb[0]) * (gb[3] - gb[1])
        best_contain = 0.0
        best_iou = 0.0
        for cb in raw_cells:
            ix = max(0, min(cb[2], gb[2]) - max(cb[0], gb[0]))
            iy = max(0, min(cb[3], gb[3]) - max(cb[1], gb[1]))
            inter = ix * iy
            if inter > 0:
                contain = inter / gt_area if gt_area > 0 else 0
                cell_area = (cb[2] - cb[0]) * (cb[3] - cb[1])
                iou = inter / (cell_area + gt_area - inter)
                if contain > best_contain: best_contain = contain
                if iou > best_iou: best_iou = iou
        if best_contain >= 0.8: contained_count += 1
        if best_iou >= 0.25: iou25_count += 1
    print(f"{img_path.stem}: GT={len(gt_boxes)}, RawCells={len(raw_cells)} | Containment>=80%: {contained_count/len(gt_boxes):.3f} | IoU>=0.25: {iou25_count/len(gt_boxes):.3f}")
