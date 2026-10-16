import csv
import numpy as np

with open("d:/RRC/major_project_csvtu/splerge_per_table_metrics.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

raw_recs = [float(r["raw_rec_25"]) for r in rows]
post_recs = [float(r["post_rec_25"]) for r in rows]
raw_rec50s = [float(r["raw_rec_50"]) for r in rows]

print("--- RAW RECALL @ 0.25 DISTRIBUTION ---")
print(f"Count: {len(raw_recs)}")
print(f"Mean: {np.mean(raw_recs):.4f}")
print(f"Median: {np.median(raw_recs):.4f}")
print(f"Min: {np.min(raw_recs):.4f}")
print(f"Max: {np.max(raw_recs):.4f}")
print(f"Std: {np.std(raw_recs):.4f}")
print(f"Quantiles [25%, 50%, 75%, 90%, 95%]: {np.percentile(raw_recs, [25, 50, 75, 90, 95])}")

print("\nTop 5 tables by raw recall @ 0.25:")
top5 = sorted(rows, key=lambda r: float(r["raw_rec_25"]), reverse=True)[:5]
for r in top5:
    print(f"  {r['table_id']}: GT={r['gt_count']}, RawCells={r['raw_cells']}, RawRec25={r['raw_rec_25']}, PostRec25={r['post_rec_25']}")

print("\nBottom 5 tables by raw recall @ 0.25:")
bot5 = sorted(rows, key=lambda r: float(r["raw_rec_25"]))[:5]
for r in bot5:
    print(f"  {r['table_id']}: GT={r['gt_count']}, RawCells={r['raw_cells']}, RawRec25={r['raw_rec_25']}, PostRec25={r['post_rec_25']}")

# Let's also check zero-recall tables:
zero_raw = [r for r in rows if float(r["raw_rec_25"]) == 0.0]
print(f"\nTables with 0.0% raw recall @ 0.25: {len(zero_raw)} / 100")
for r in zero_raw[:5]:
    print(f"  {r['table_id']}: GT={r['gt_count']}, RawCells={r['raw_cells']}")
