import sys
import os
import json
import csv
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path

splerge_path = Path("d:/RRC/major_project_csvtu/AdvancedLiterateMachinery/DocumentUnderstanding/SPLERGE")
sys.path.insert(0, str(splerge_path))

from libs.split_model_hf import SplitModel
import libs.merge_utility as mut
import libs.GTElement as gc
import merge

def get_middle_of_groups(binary_array):
    result = np.zeros_like(binary_array)
    i = 0
    n = len(binary_array)
    while i < n:
        if binary_array[i] == 1:
            start = i
            while i < n and binary_array[i] == 1:
                i += 1
            end = i - 1
            middle = (start + end) // 2
            result[middle] = 1
        else:
            i += 1
    return result

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def count_internal_overlaps(boxes):
    overlaps = 0
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            b1, b2 = boxes[i], boxes[j]
            ix = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
            iy = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
            if ix > 0 and iy > 0:
                overlaps += 1
    return overlaps

def evaluate_matching(pred_boxes, gt_boxes, thresh=0.5):
    matched_gt = set()
    for pb in pred_boxes:
        best_iou = 0
        best_j = -1
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= thresh:
            matched_gt.add(best_j)
    tp = len(matched_gt)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    prec = tp / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
    rec = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return matched_gt, tp, fp, fn, prec, rec, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_weight_path = splerge_path / "model_out" / "split_model.pth"
    model = SplitModel().to(device)
    checkpoint = torch.load(model_weight_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((960, 960)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dir = Path("d:/RRC/major_project_csvtu/pubtab_samples_tokens/images")
    json_dir = Path("d:/RRC/major_project_csvtu/pubtab_samples_tokens/iocr_jsons")
    test_images = sorted(list(test_dir.glob("*.png")))

    print(f"Starting SPLERGE 100-table evaluation on {len(test_images)} tables...")

    per_table_rows = []
    
    total_gt = 0
    total_raw_cells = 0
    total_raw_overlaps = 0
    total_raw_tp50 = 0
    total_raw_tp25 = 0

    total_post_cells = 0
    total_post_overlaps = 0
    total_post_tp50 = 0
    total_post_tp25 = 0

    total_destroyed_tp50 = 0
    total_destroyed_tp25 = 0

    tables_tp_decreased = 0
    tables_tp_increased = 0
    tables_tp_unchanged = 0

    for idx, img_path in enumerate(test_images):
        json_path = json_dir / (img_path.stem + "_iocr.json")
        if not json_path.exists():
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        gt_tokens = data["pages"][0].get("tokens", [])
        gt_boxes = [tok["bbox"] for tok in gt_tokens]
        raw_ocr = [[tok["id"], tok["text"], b[0], b[1], b[2], b[3]] for tok, b in zip(gt_tokens, gt_boxes)]

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            h_pred, v_pred = model(img_tensor)
            h_pred = h_pred.repeat_interleave(2, dim=1).squeeze(0).cpu().numpy()
            v_pred = v_pred.repeat_interleave(2, dim=1).squeeze(0).cpu().numpy()

        h_bin = (h_pred > 0.5).astype(float)
        v_bin = (v_pred > 0.5).astype(float)
        h_clean = get_middle_of_groups(h_bin)
        v_clean = get_middle_of_groups(v_bin)

        h_splits = np.where(h_clean == 1)[0]
        v_splits = np.where(v_clean == 1)[0]

        row_separators = [int(y * orig_h / 960.0) for y in h_splits if 5 < int(y * orig_h / 960.0) < orig_h - 5]
        col_separators = [int(x * orig_w / 960.0) for x in v_splits if 5 < int(x * orig_w / 960.0) < orig_w - 5]

        # 1. Raw Split Grid
        _table_raw = gc.Table(0, 0, orig_w, orig_h)
        for r in row_separators:
            _table_raw.gtRows.append(gc.Row(0, r, orig_w))
        for c in col_separators:
            _table_raw.gtCols.append(gc.Column(c, 0, orig_h))
        _table_raw.evaluateCells()

        raw_cells = [[c.x1, c.y1, c.x2, c.y2] for row in _table_raw.gtCells for c in row if not c.dontCare]
        raw_overlaps = count_internal_overlaps(raw_cells)

        raw_m50, raw_tp50, raw_fp50, raw_fn50, raw_p50, raw_r50, raw_f1_50 = evaluate_matching(raw_cells, gt_boxes, thresh=0.5)
        raw_m25, raw_tp25, raw_fp25, raw_fn25, raw_p25, raw_r25, raw_f1_25 = evaluate_matching(raw_cells, gt_boxes, thresh=0.25)

        # 2. Merge Pipeline
        ocr_data = mut.clean_ocr_data(raw_ocr)
        ocr_data = mut.make_sentences(ocr_data)

        _table_merge = gc.Table(0, 0, orig_w, orig_h)
        for r in row_separators:
            _table_merge.gtRows.append(gc.Row(0, r, orig_w))
        for c in col_separators:
            _table_merge.gtCols.append(gc.Column(c, 0, orig_h))
        _table_merge.evaluateCells()

        merge.col_merging(_table_merge, ocr_data)
        merge.row_merging(_table_merge, ocr_data)
        merge.remove_redundant_seperators(_table_merge, ocr_data, 0.2)
        merge.remove_redundant_seperators(_table_merge, ocr_data, 0.4)
        _table_merge.merge_header_v2(ocr_data)
        _table_merge.evaluateCells()

        post_cells = [[c.x1, c.y1, c.x2, c.y2] for row in _table_merge.gtCells for c in row if not c.dontCare]
        post_overlaps = count_internal_overlaps(post_cells)

        post_m50, post_tp50, post_fp50, post_fn50, post_p50, post_r50, post_f1_50 = evaluate_matching(post_cells, gt_boxes, thresh=0.5)
        post_m25, post_tp25, post_fp25, post_fn25, post_p25, post_r25, post_f1_25 = evaluate_matching(post_cells, gt_boxes, thresh=0.25)

        # Cells that were matched in raw grid but lost after merge
        destroyed_50 = len([j for j in raw_m50 if j not in post_m50])
        destroyed_25 = len([j for j in raw_m25 if j not in post_m25])

        total_gt += len(gt_boxes)
        total_raw_cells += len(raw_cells)
        total_raw_overlaps += raw_overlaps
        total_raw_tp50 += raw_tp50
        total_raw_tp25 += raw_tp25

        total_post_cells += len(post_cells)
        total_post_overlaps += post_overlaps
        total_post_tp50 += post_tp50
        total_post_tp25 += post_tp25

        total_destroyed_tp50 += destroyed_50
        total_destroyed_tp25 += destroyed_25

        if post_tp25 < raw_tp25:
            tables_tp_decreased += 1
        elif post_tp25 > raw_tp25:
            tables_tp_increased += 1
        else:
            tables_tp_unchanged += 1

        per_table_rows.append({
            "table_id": img_path.stem,
            "gt_count": len(gt_boxes),
            "raw_cells": len(raw_cells),
            "raw_overlaps": raw_overlaps,
            "raw_tp_50": raw_tp50,
            "raw_prec_50": round(raw_p50, 4),
            "raw_rec_50": round(raw_r50, 4),
            "raw_f1_50": round(raw_f1_50, 4),
            "raw_tp_25": raw_tp25,
            "raw_prec_25": round(raw_p25, 4),
            "raw_rec_25": round(raw_r25, 4),
            "raw_f1_25": round(raw_f1_25, 4),
            "post_cells": len(post_cells),
            "post_overlaps": post_overlaps,
            "post_tp_50": post_tp50,
            "post_prec_50": round(post_p50, 4),
            "post_rec_50": round(post_r50, 4),
            "post_f1_50": round(post_f1_50, 4),
            "post_tp_25": post_tp25,
            "post_prec_25": round(post_p25, 4),
            "post_rec_25": round(post_r25, 4),
            "post_f1_25": round(post_f1_25, 4),
            "delta_cells": len(post_cells) - len(raw_cells),
            "delta_tp_50": post_tp50 - raw_tp50,
            "delta_tp_25": post_tp25 - raw_tp25,
            "destroyed_tp_50": destroyed_50,
            "destroyed_tp_25": destroyed_25
        })

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(test_images)} tables...")

    # Write CSV
    csv_path = Path("d:/RRC/major_project_csvtu/splerge_per_table_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_table_rows)

    # Compute Global Summary
    raw_prec50_glob = total_raw_tp50 / total_raw_cells if total_raw_cells > 0 else 0
    raw_rec50_glob = total_raw_tp50 / total_gt if total_gt > 0 else 0
    raw_f1_50_glob = 2 * raw_prec50_glob * raw_rec50_glob / (raw_prec50_glob + raw_rec50_glob) if (raw_prec50_glob + raw_rec50_glob) > 0 else 0

    raw_prec25_glob = total_raw_tp25 / total_raw_cells if total_raw_cells > 0 else 0
    raw_rec25_glob = total_raw_tp25 / total_gt if total_gt > 0 else 0
    raw_f1_25_glob = 2 * raw_prec25_glob * raw_rec25_glob / (raw_prec25_glob + raw_rec25_glob) if (raw_prec25_glob + raw_rec25_glob) > 0 else 0

    post_prec50_glob = total_post_tp50 / total_post_cells if total_post_cells > 0 else 0
    post_rec50_glob = total_post_tp50 / total_gt if total_gt > 0 else 0
    post_f1_50_glob = 2 * post_prec50_glob * post_rec50_glob / (post_prec50_glob + post_rec50_glob) if (post_prec50_glob + post_rec50_glob) > 0 else 0

    post_prec25_glob = total_post_tp25 / total_post_cells if total_post_cells > 0 else 0
    post_rec25_glob = total_post_tp25 / total_gt if total_gt > 0 else 0
    post_f1_25_glob = 2 * post_prec25_glob * post_rec25_glob / (post_prec25_glob + post_rec25_glob) if (post_prec25_glob + post_rec25_glob) > 0 else 0

    summary = {
        "num_tables": len(per_table_rows),
        "total_gt_cells": total_gt,
        "raw_grid": {
            "total_cells": total_raw_cells,
            "total_overlaps": total_raw_overlaps,
            "overlap_rate": round(total_raw_overlaps / total_raw_cells, 6) if total_raw_cells > 0 else 0.0,
            "iou_05": {
                "tp": total_raw_tp50,
                "precision": round(raw_prec50_glob, 4),
                "recall": round(raw_rec50_glob, 4),
                "f1": round(raw_f1_50_glob, 4)
            },
            "iou_025": {
                "tp": total_raw_tp25,
                "precision": round(raw_prec25_glob, 4),
                "recall": round(raw_rec25_glob, 4),
                "f1": round(raw_f1_25_glob, 4)
            }
        },
        "post_merge": {
            "total_cells": total_post_cells,
            "total_overlaps": total_post_overlaps,
            "overlap_rate": round(total_post_overlaps / total_post_cells, 6) if total_post_cells > 0 else 0.0,
            "iou_05": {
                "tp": total_post_tp50,
                "precision": round(post_prec50_glob, 4),
                "recall": round(post_rec50_glob, 4),
                "f1": round(post_f1_50_glob, 4)
            },
            "iou_025": {
                "tp": total_post_tp25,
                "precision": round(post_prec25_glob, 4),
                "recall": round(post_rec25_glob, 4),
                "f1": round(post_f1_25_glob, 4)
            }
        },
        "destructive_repair_impact": {
            "cell_reduction": total_post_cells - total_raw_cells,
            "delta_tp_50": total_post_tp50 - total_raw_tp50,
            "delta_tp_25": total_post_tp25 - total_raw_tp25,
            "gross_destroyed_tp_50": total_destroyed_tp50,
            "gross_destroyed_tp_25": total_destroyed_tp25,
            "destruction_rate_25": round(total_destroyed_tp25 / total_raw_tp25, 4) if total_raw_tp25 > 0 else 0.0,
            "tables_tp_decreased": tables_tp_decreased,
            "tables_tp_increased": tables_tp_increased,
            "tables_tp_unchanged": tables_tp_unchanged
        }
    }

    summary_path = Path("d:/RRC/major_project_csvtu/splerge_audit_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n================== SPLERGE 100-TABLE AUDIT COMPLETE ==================")
    print(f"Evaluated {len(per_table_rows)} tables ({total_gt} GT cells).")
    print(f"Raw Split Grid:   {total_raw_cells} cells | Overlaps: {total_raw_overlaps} | TP@0.25: {total_raw_tp25} ({raw_rec25_glob*100:.1f}% Rec)")
    print(f"Post-Merge:       {total_post_cells} cells | Overlaps: {total_post_overlaps} | TP@0.25: {total_post_tp25} ({post_rec25_glob*100:.1f}% Rec)")
    print(f"Destructive Loss: Gross {total_destroyed_tp25} TPs destroyed ({summary['destructive_repair_impact']['destruction_rate_25']*100:.1f}% of all detected cells wiped out)")
    print(f"Tables Affected:  {tables_tp_decreased} tables lost TPs, {tables_tp_increased} gained TPs, {tables_tp_unchanged} unchanged.")
    print(f"Saved: {csv_path} and {summary_path}")

if __name__ == "__main__":
    main()
