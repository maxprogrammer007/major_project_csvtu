import sys
import os
import json
import time
from pathlib import Path

# Add workspace and SPLERGE paths
BASE_DIR = Path("d:/RRC/major_project_csvtu")
SPLERGE_PATH = BASE_DIR / "AdvancedLiterateMachinery/DocumentUnderstanding/SPLERGE"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SPLERGE_PATH))

import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
import cv2
from PIL import Image

import libs.GTElement as gc
import libs.merge_utility as mut
import merge as merge_mod

from teds_metric import TEDS
import grits

# Load cached grid coordinates and GT HTML
grid_cache = json.load(open(BASE_DIR / "scratch/splerge_100_tables_grid_cache.json"))
gt_html_cache = json.load(open(BASE_DIR / "scratch/pubtabnet_100_tables_gt_html.json"))

print(f"Loaded {len(grid_cache)} cached grids and {len(gt_html_cache)} GT HTMLs.")
test_key = list(grid_cache.keys())[0]
print(f"Sample table: {test_key}")
