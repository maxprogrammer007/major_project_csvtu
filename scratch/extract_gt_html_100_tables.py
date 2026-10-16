import json
import ast
from pathlib import Path
from datasets import load_dataset

CACHE_FILE = Path("scratch/splerge_100_tables_grid_cache.json")
with open(CACHE_FILE, "r") as f:
    grid_cache = json.load(f)

# Extract integer imgids
imgid_to_stem = {}
for stem in grid_cache.keys():
    # stem is e.g. pubtabnet_tok_550868
    imgid = int(stem.replace("pubtabnet_tok_", ""))
    imgid_to_stem[imgid] = stem

print(f"Targeting {len(imgid_to_stem)} tables from PubTabNet...")

def parse_html(html_str):
    try:
        return json.loads(html_str)
    except Exception:
        try:
            return ast.literal_eval(html_str)
        except Exception:
            return {}

def structure_to_html(structure_tokens, cells):
    """
    Reconstruct full HTML table string from PubTabNet structure tokens and cells.
    """
    html_out = []
    cell_idx = 0
    for tok in structure_tokens:
        if tok == "<td></td>":
            if cell_idx < len(cells):
                cell_text = "".join(cells[cell_idx].get("tokens", []))
                html_out.append(f"<td>{cell_text}</td>")
                cell_idx += 1
            else:
                html_out.append("<td></td>")
        elif tok.startswith("<td"):
            # Spanning cell tag e.g. <td colspan="2">
            if cell_idx < len(cells):
                cell_text = "".join(cells[cell_idx].get("tokens", []))
                # Replace ending > with >cell_text</td> if self-closing or follow structure
                html_out.append(f"{tok}{cell_text}</td>")
                cell_idx += 1
            else:
                html_out.append(f"{tok}</td>")
        elif tok == "</td>":
            continue
        else:
            html_out.append(tok)
    return "".join(html_out)

found = {}
# Stream validation split first
print("Checking validation split...")
ds_val = load_dataset("apoidea/pubtabnet-html", split="validation", streaming=True)
for item in ds_val:
    imgid = item.get("imgid")
    if imgid in imgid_to_stem:
        stem = imgid_to_stem[imgid]
        raw_html = item.get("html", "{}")
        parsed = parse_html(raw_html)
        found[stem] = {
            "imgid": imgid,
            "html_data": parsed,
            "split": "validation"
        }
        if len(found) == len(imgid_to_stem):
            break

print(f"Found {len(found)} in validation split.")

if len(found) < len(imgid_to_stem):
    missing_ids = set(imgid_to_stem.keys()) - {d["imgid"] for d in found.values()}
    print(f"Checking train split for {len(missing_ids)} missing tables...")
    ds_train = load_dataset("apoidea/pubtabnet-html", split="train", streaming=True)
    for item in ds_train:
        imgid = item.get("imgid")
        if imgid in missing_ids:
            stem = imgid_to_stem[imgid]
            raw_html = item.get("html", "{}")
            parsed = parse_html(raw_html)
            found[stem] = {
                "imgid": imgid,
                "html_data": parsed,
                "split": "train"
            }
            missing_ids.discard(imgid)
            if not missing_ids:
                break

print(f"Total found across splits: {len(found)} / {len(imgid_to_stem)}")
OUT_CACHE = Path("scratch/pubtabnet_100_tables_gt_html.json")
with open(OUT_CACHE, "w", encoding="utf-8") as f:
    json.dump(found, f)

print(f"Saved ground truth HTML data to {OUT_CACHE}")
