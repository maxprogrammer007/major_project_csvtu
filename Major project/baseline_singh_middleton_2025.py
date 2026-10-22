"""
Reimplementation of Singh & Middleton (IJDAR 2025) Geometric Completion Baseline:
"Tabular Context-Aware Optical Character Recognition and Tabular Data Reconstruction for Historical Records"
IJDAR vol. 28, pp. 357-376, 2025. DOI: 10.1007/s10032-025-00543-9.

Mechanism (Section 3.2 verbatim):
"infer missing cells based on the horizontal and vertical alignment of detected positive cells.
 This approach ensures complete table reconstruction by generating candidate cells where gaps
 are identified and aligning them with existing cells."
"""

import numpy as np
from fitz import Rect

def compute_box_iou(b1, b2):
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2])
    y1 = min(b1[3], b2[3])
    
    inter_w = max(0.0, x1 - x0)
    inter_h = max(0.0, y1 - y0)
    inter_area = inter_w * inter_h
    
    area1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def singh_middleton_reconstruction(detected_rows, detected_cols, detected_cells, span_cells=None, iou_thresh=0.3):
    """
    Infers missing cells from the alignment grid of detected positive rows and columns.
    
    Args:
        detected_rows: list of row dicts with 'bbox': [x0, y0, x1, y1]
        detected_cols: list of col dicts with 'bbox': [x0, y0, x1, y1]
        detected_cells: list of existing cell dicts with 'bbox': [x0, y0, x1, y1]
        span_cells: optional list of spanning cell bboxes (to measure span violation)
        iou_thresh: overlap threshold to consider a grid vacancy already occupied
        
    Returns:
        inferred_cells: list of newly generated candidate cells
        full_reconstructed_cells: existing cells + inferred cells
    """
    if len(detected_rows) == 0 or len(detected_cols) == 0:
        return [], detected_cells
        
    inferred_cells = []
    
    # Sort rows top-to-bottom, cols left-to-right
    sorted_rows = sorted(detected_rows, key=lambda r: r['bbox'][1])
    sorted_cols = sorted(detected_cols, key=lambda c: c['bbox'][0])
    
    existing_boxes = [c['bbox'] for c in detected_cells]
    
    # Form Cartesian product grid
    for r_idx, row in enumerate(sorted_rows):
        r_box = row['bbox']
        for c_idx, col in enumerate(sorted_cols):
            c_box = col['bbox']
            
            # Intersection coordinate box
            x0 = c_box[0]
            x1 = c_box[2]
            y0 = r_box[1]
            y1 = r_box[3]
            
            if x1 <= x0 or y1 <= y0:
                continue
                
            candidate_box = [x0, y0, x1, y1]
            
            # Check if this vacancy is already occupied by an existing detected cell
            max_iou = 0.0
            for eb in existing_boxes:
                iou_val = compute_box_iou(candidate_box, eb)
                if iou_val > max_iou:
                    max_iou = iou_val
                    
            if max_iou < iou_thresh:
                # Identified a gap! Inferred missing cell
                is_span_violation = False
                if span_cells:
                    cand_rect = Rect(candidate_box)
                    for sc in span_cells:
                        sc_rect = Rect(sc['bbox'] if isinstance(sc, dict) else sc)
                        if sc_rect.intersect(cand_rect).get_area() / cand_rect.get_area() > 0.5:
                            is_span_violation = True
                            break
                            
                inferred_cells.append({
                    'bbox': candidate_box,
                    'row_num': r_idx,
                    'col_num': c_idx,
                    'inferred_by': 'singh_middleton_2025',
                    'span_violation': is_span_violation
                })
                
    full_reconstructed = list(detected_cells) + inferred_cells
    return inferred_cells, full_reconstructed

if __name__ == "__main__":
    # Smoke test on dummy table
    test_rows = [{'bbox': [10, 10, 100, 30]}, {'bbox': [10, 35, 100, 55]}]
    test_cols = [{'bbox': [10, 10, 50, 55]}, {'bbox': [55, 10, 100, 55]}]
    # Suppose top-right cell (row 0, col 1) was missed by base detector
    detected_cells = [
        {'bbox': [10, 10, 50, 30]},  # row 0, col 0
        {'bbox': [10, 35, 50, 55]},  # row 1, col 0
        {'bbox': [55, 35, 100, 55]}, # row 1, col 1
    ]
    inferred, full = singh_middleton_reconstruction(test_rows, test_cols, detected_cells)
    print("Inferred missing cells:", len(inferred))
    print("Inferred box:", inferred[0]['bbox'])
    assert len(inferred) == 1
    assert inferred[0]['bbox'] == [55, 10, 100, 30]
    print("Smoke test passed successfully!")
