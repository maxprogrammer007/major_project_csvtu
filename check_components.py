import networkx as nx
from pathlib import Path
from PIL import Image
import torch
from transformers import TableTransformerForObjectDetection, DetrImageProcessor

from optimize_tatr_consistency import object2_overlap

images_dir = Path('pubtables_test_sample/images')
model_id = 'microsoft/table-transformer-structure-recognition-v1.1-all'
processor = DetrImageProcessor.from_pretrained(model_id)
model = TableTransformerForObjectDetection.from_pretrained(model_id)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
SIZE = {'shortest_edge': 800, 'longest_edge': 1000}
SCORE_THRESH = 0.5

cc_sizes = []
image_paths = sorted(images_dir.glob('*.jpg'))
print(f'Processing {len(image_paths)} tables for components')

for img_path in image_paths:
    img = Image.open(img_path).convert('RGB')
    inputs = processor(images=img, return_tensors='pt', size=SIZE)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([img.size[::-1]]).to(device)
    results = processor.post_process_object_detection(outputs, threshold=SCORE_THRESH, target_sizes=target_sizes)[0]
    
    labels = results['labels'].tolist()
    scores = results['scores'].tolist()
    boxes = results['boxes'].tolist()
    
    raw_rows = []
    for b, s, l in zip(boxes, scores, labels):
        if model.config.id2label[l] == 'table row':
            raw_rows.append({'bbox': [float(c) for c in b], 'score': float(s)})
            
    n = len(raw_rows)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            r1, r2 = object2_overlap(raw_rows[i]['bbox'], raw_rows[j]['bbox'])
            if r1 > 0.5 or r2 > 0.5:
                edges.append((i, j))
                
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    
    cc_sizes.extend([len(c) for c in nx.connected_components(G)])

print(f'Total components: {len(cc_sizes)}, >2 nodes: {sum(1 for s in cc_sizes if s > 2)}')
print(f'Fraction >2 nodes: {sum(1 for s in cc_sizes if s > 2) / len(cc_sizes):.4f}')
