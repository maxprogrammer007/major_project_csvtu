import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import grits
from teds_metric import TEDS

h1 = '<table><tr><td colspan="2">Header</td></tr><tr><td>A</td><td>B</td></tr></table>'
res = grits.grits_from_html(h1, h1)
print("Self GriTS:", res)

teds = TEDS(structure_only=True)
print("Self TEDS-S:", teds.evaluate(h1, h1))
