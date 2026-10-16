import json
from pathlib import Path
from lxml import html

table_id = "pubtabnet_tok_550868"
meta_path = Path("pubtab_samples_tokens/iocr_jsons") / f"{table_id}_iocr.json"
with open(meta_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Table: {table_id}")
gt_html = "".join(data['html']['structure']['tokens'])
print(f"GT HTML length: {len(gt_html)}")

root = html.fromstring(gt_html)
gt_trs = root.xpath('//tr')
print(f"GT tr count: {len(gt_trs)}")
for i, tr in enumerate(gt_trs):
    tds = tr.xpath('./td')
    rowspans = [td.get('rowspan', '1') for td in tds]
    colspans = [td.get('colspan', '1') for td in tds]
    # show first few
    if i < 5 or i >= len(gt_trs) - 2:
        print(f"  GT row {i:02d}: {len(tds)} tds (rowspans: {rowspans}, colspans: {colspans})")

