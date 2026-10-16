import json
from pathlib import Path
import os
import sys

# Need to import our optimized solver to see what candidates are generated
sys.path.append('d:/RRC/major_project_csvtu')
from optimize_splerge_consistency import evaluate_table_optimization

def investigate_table(table_id):
    gt_path = Path(f'd:/RRC/major_project_csvtu/pubtabnet_subset/pubtabnet_tok_{table_id}_iocr.json')
    if not gt_path.exists():
        print(f"GT not found: {gt_path}")
        return
        
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
        
    # Run the optimization logic to get exactly what we evaluated
    result = evaluate_table_optimization(table_id, gt_data)
    
    # We want to know which GT cells were matched by Raw, Heuristic, and H2.
    # Actually, we can just look at the GT cells directly and see their shapes in terms of the grid.
    
    # Let's extract the grid and candidate pool.
    print(f"\n--- Investigating {table_id} ---")
    print(f"H2 TP: {result['opt_metrics']['tp']} / GT: {result['gt_count']}")
    print(f"Raw TP: {result['raw_metrics']['tp']} / GT: {result['gt_count']}")
    print(f"Heuristic TP: {result['heuristic_metrics']['tp']} / GT: {result['gt_count']}")
    
    # We want to identify the specific GT cells that were missed by H2 but hit by Raw.
    # Wait, H2 recovered 632 out of 634 cells destroyed by heuristic.
    # This means Raw hit them, Heuristic missed them, and H2 missed 2 of them.
    # Actually, were they hit by Raw? If Heuristic destroyed them, it means they were present in Raw.
    # Let's recompute the exact matching.
    
    # We need to compute IoU for the generated boxes against GT.
    from optimize_splerge_consistency import compute_iou
    
    def get_matched_gt_indices(pred_boxes, gt_boxes, iou_thresh=0.25):
        matched = set()
        for p in pred_boxes:
            for i, g in enumerate(gt_boxes):
                if compute_iou(p, g) >= iou_thresh:
                    matched.add(i)
        return matched
        
    raw_boxes = result['raw_boxes']
    opt_boxes = result['opt_boxes']
    gt_boxes = result['gt_boxes']
    
    raw_matched = get_matched_gt_indices(raw_boxes, gt_boxes)
    opt_matched = get_matched_gt_indices(opt_boxes, gt_boxes)
    
    missed_by_opt = raw_matched - opt_matched
    print(f"GT indices missed by H2 (but hit by Raw): {missed_by_opt}")
    
    for idx in missed_by_opt:
        gt_box = gt_boxes[idx]
        print(f"  GT Box: {gt_box}")
        # Find which grid tiles intersect this GT box
        from optimize_splerge_consistency import generate_grid_tiles
        # We need the split projections. We can just load the raw output.
        raw_output_path = Path(f'd:/RRC/major_project_csvtu/splerge_outputs/pubtabnet_tok_{table_id}_iocr_raw_split.json')
        with open(raw_output_path, 'r') as f:
            raw_output = json.load(f)
        
        y_cuts = raw_output['row_cuts']
        x_cuts = raw_output['col_cuts']
        grid_tiles = generate_grid_tiles(y_cuts, x_cuts)
        
        # Count how many tiles it spans
        span_r_min, span_r_max = float('inf'), -float('inf')
        span_c_min, span_c_max = float('inf'), -float('inf')
        
        for (r, c), tile in grid_tiles.items():
            if compute_iou(tile, gt_box) > 0.05:  # meaningful overlap
                span_r_min = min(span_r_min, r)
                span_r_max = max(span_r_max, r)
                span_c_min = min(span_c_min, c)
                span_c_max = max(span_c_max, c)
                
        height = (span_r_max - span_r_min + 1) if span_r_max >= span_r_min else 0
        width = (span_c_max - span_c_min + 1) if span_c_max >= span_c_min else 0
        
        print(f"  Estimated Grid Span: height={height}, width={width}")
        if height > 3 or width > 2:
            print("  -> EXCEEDS CANDIDATE POOL CAP (height<=3, width<=2)")
        else:
            print("  -> WITHIN CAP (Ambiguity failure)")

if __name__ == '__main__':
    investigate_table('586722')
    investigate_table('711083')
