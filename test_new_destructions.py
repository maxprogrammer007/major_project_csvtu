import torch
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import copy

from transformers import TableTransformerForObjectDetection, DetrImageProcessor
from optimize_tatr_consistency import object2_overlap, solve_mwis, match_boxes
import tatr_postprocess as postprocess

model_id = 'microsoft/table-transformer-structure-recognition-v1.1-all'
processor = DetrImageProcessor.from_pretrained(model_id)
model = TableTransformerForObjectDetection.from_pretrained(model_id)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

tables = ['PMC5762819_table_1', 'PMC5897469_table_0']

for t in tables:
    img_path = Path(f'pubtables_test_sample/images/{t}.jpg')
    xml_path = Path(f'pubtables_test_sample/annotations/{t}.xml')
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt_rows = []
    for obj in root.findall('object'):
        if obj.find('name').text == 'table row':
            gt_rows.append([float(obj.find('bndbox').find(k).text) for k in ['xmin', 'ymin', 'xmax', 'ymax']])
            
    img = Image.open(img_path).convert('RGB')
    inputs = processor(images=img, return_tensors='pt', size={'shortest_edge': 800, 'longest_edge': 1000})
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=torch.tensor([img.size[::-1]]).to(device))[0]
    labels = results['labels'].tolist()
    scores = results['scores'].tolist()
    boxes = results['boxes'].tolist()
    
    raw_rows = []
    for b, s, l in zip(boxes, scores, labels):
        if model.config.id2label[l] == 'table row':
            raw_rows.append({'bbox': [float(coord) for coord in b], 'score': float(s)})
            
    med_h = np.median([r['bbox'][3] - r['bbox'][1] for r in raw_rows])
    
    mod_rows = []
    for r in raw_rows:
        h = r['bbox'][3] - r['bbox'][1]
        dist = abs(h - med_h) / med_h
        score = max(0.01, 1.0 - dist)
        mod_rows.append({'bbox': r['bbox'], 'score': score, 'raw_score': r['score'], 'h': h})
        
    opt_rows, _ = solve_mwis(mod_rows, object2_overlap, threshold=0.5)
    heur_rows = postprocess.refine_rows(copy.deepcopy(raw_rows), tokens=[], score_threshold=0.5)
    
    heur_tp, _, _ = match_boxes([r['bbox'] for r in heur_rows], gt_rows, iou_thresh=0.5)
    opt_tp, _, _ = match_boxes([r['bbox'] for r in opt_rows], gt_rows, iou_thresh=0.5)
    
    heur_tp_bboxes = [heur_rows[i]['bbox'] for i, j, _ in heur_tp]
    opt_tp_bboxes = [opt_rows[i]['bbox'] for i, j, _ in opt_tp]
    
    destroyed = [r for r in mod_rows if r['bbox'] in heur_tp_bboxes and r['bbox'] not in opt_tp_bboxes]
    
    print(f'\n--- {t} ---')
    print(f'Median Height: {med_h:.2f}')
    
    for r in destroyed:
        print(f"Destroyed Genuine Row: Height={r['h']:.2f}, RawScore={r['raw_score']:.4f}, HeightObjScore={r['score']:.4f}")
        
        conflicts = []
        for o in opt_rows:
            r1, r2 = object2_overlap(r['bbox'], o['bbox'])
            if (r1 > 0.5 or r2 > 0.5) and o['bbox'] != r['bbox']:
                conflicts.append(o)
                
        for c in conflicts:
            c_h = c['bbox'][3] - c['bbox'][1]
            print(f"  Suppressed by: Height={c_h:.2f}, RawScore={c['raw_score']:.4f}, HeightObjScore={c['score']:.4f}")
