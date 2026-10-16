import pandas as pd

df = pd.read_csv('metric_suite_results.csv')
diff = df.copy()
diff['tp_gain'] = diff['h2ocr_tp'] - diff['h1ocr_tp']
diff['teds_drop'] = diff['h1ocr_teds_s'] - diff['h2ocr_teds_s']
cand = diff[(diff['tp_gain'] >= 5) & (diff['teds_drop'] > 0.05)].sort_values('teds_drop', ascending=False)
print(f"Found {len(cand)} tables with TP gain >= 5 and H1 TEDS-S higher by > 0.05:")
for idx, r in cand.head(10).iterrows():
    print(f"{r['table_id']}: GT cells={r['gt_cells']}, Raw TP={r['raw_tp']}, H1 TP={r['h1ocr_tp']}, H2 TP={r['h2ocr_tp']} (gain +{r['tp_gain']}), H1 TEDS={r['h1ocr_teds_s']:.4f}, H2 TEDS={r['h2ocr_teds_s']:.4f} (diff {r['teds_drop']:+.4f})")
