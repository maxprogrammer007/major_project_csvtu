import json
import numpy as np
import scipy.optimize
from pathlib import Path

CACHE_FILE = Path('scratch/tableformer_100_tables_cache.json')
with open(CACHE_FILE, 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)} tables from cache.")

def compute_intersection_area(b1, b2):
    x0, y0 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x1, y1 = min(b1[2], b2[2]), min(b1[3], b2[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)

fractional_count = 0
total_solved = 0
linprog_vs_milp_diff = 0

for item in data:
    raw_boxes = item['raw_boxes']
    tokens = item['tokens']
    n = len(raw_boxes)
    if n == 0: continue

    weights = np.zeros(n)
    for i, b in enumerate(raw_boxes):
        box_w = max(0.0, b[2] - b[0])
        box_h = max(0.0, b[3] - b[1])
        box_area = box_w * box_h
        if box_area <= 0:
            weights[i] = 0.0
            continue
        inter_sum = sum(compute_intersection_area(b, t['bbox']) for t in tokens)
        weights[i] = (inter_sum / box_area) + 0.01

    c = -weights
    conflicts = []
    for i in range(n):
        for j in range(i + 1, n):
            inter = compute_intersection_area(raw_boxes[i], raw_boxes[j])
            if inter > 0:
                area_i = max(0.0, raw_boxes[i][2] - raw_boxes[i][0]) * max(0.0, raw_boxes[i][3] - raw_boxes[i][1])
                area_j = max(0.0, raw_boxes[j][2] - raw_boxes[j][0]) * max(0.0, raw_boxes[j][3] - raw_boxes[j][1])
                min_area = min(area_i, area_j)
                if min_area > 0 and (inter / min_area) > 0.1:
                    conflicts.append((i, j))

    if not conflicts:
        continue

    total_solved += 1
    A_ub = np.zeros((len(conflicts), n))
    b_ub = np.ones(len(conflicts))
    for k, (i, j) in enumerate(conflicts):
        A_ub[k, i] = 1
        A_ub[k, j] = 1

    # linprog with integrality=1
    res_lp = scipy.optimize.linprog(
        c, A_ub=A_ub, b_ub=b_ub,
        bounds=(0, 1), integrality=1,
        method='highs'
    )
    
    # milp
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import csc_matrix
    constraints = LinearConstraint(csc_matrix(A_ub), ub=b_ub)
    integrality = np.ones(n)
    bounds = Bounds(0, 1)
    res_milp = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)

    # Check integrality of res_lp.x
    frac = np.any((res_lp.x > 1e-5) & (res_lp.x < 1 - 1e-5))
    if frac:
        fractional_count += 1
        print(f"Table {item['stem']}: res_lp has fractional values: {res_lp.x[np.where((res_lp.x > 1e-5) & (res_lp.x < 1 - 1e-5))[0]]}")

    # Check match between linprog and milp
    lp_sel = (res_lp.x > 0.5)
    milp_sel = (res_milp.x > 0.5)
    if not np.array_equal(lp_sel, milp_sel):
        linprog_vs_milp_diff += 1
        print(f"Table {item['stem']}: linprog vs milp difference in selection!")

print(f"\nVerification Results:")
print(f"Total tables with conflicts solved: {total_solved}")
print(f"Tables with fractional linprog solution: {fractional_count}")
print(f"Tables with discrepancy between linprog(integrality=1) and milp: {linprog_vs_milp_diff}")
