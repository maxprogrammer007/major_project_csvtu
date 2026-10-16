import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
import numpy as np

df = pd.read_csv('d:/RRC/major_project_csvtu/splerge_h2_per_table_metrics.csv')

print("--- Recall/TP (H2 vs Heuristic) ---")
wins = (df['h2_tp_25'] > df['heur_tp_25']).sum()
ties = (df['h2_tp_25'] == df['heur_tp_25']).sum()
losses = (df['h2_tp_25'] < df['heur_tp_25']).sum()
print(f"Wins: {wins}, Ties: {ties}, Losses: {losses}")

print("\n--- F1 (H2 vs Heuristic) ---")
wins_f1 = (df['h2_f1_25'] > df['heur_f1_25']).sum()
ties_f1 = (df['h2_f1_25'] == df['heur_f1_25']).sum()
losses_f1 = (df['h2_f1_25'] < df['heur_f1_25']).sum()
print(f"Wins: {wins_f1}, Ties: {ties_f1}, Losses: {losses_f1}")

print("\n--- Precision (H2 vs Heuristic) ---")
wins_p = (df['h2_prec_25'] > df['heur_prec_25']).sum()
ties_p = (df['h2_prec_25'] == df['heur_prec_25']).sum()
losses_p = (df['h2_prec_25'] < df['heur_prec_25']).sum()
print(f"Wins: {wins_p}, Ties: {ties_p}, Losses: {losses_p}")

print("\n--- Recall/TP (H2 vs Raw) ---")
wins_raw = (df['h2_tp_25'] > df['raw_tp_25']).sum()
ties_raw = (df['h2_tp_25'] == df['raw_tp_25']).sum()
losses_raw = (df['h2_tp_25'] < df['raw_tp_25']).sum()
print(f"Wins: {wins_raw}, Ties: {ties_raw}, Losses: {losses_raw}")

diff = df['h2_tp_25'] - df['raw_tp_25']
t_stat, p_val_t = ttest_rel(df['h2_tp_25'], df['raw_tp_25'])
# wilcoxon throws warning if all differences are zero, so check first
if diff.any():
    w_stat, p_val_w = wilcoxon(df['h2_tp_25'], df['raw_tp_25'])
else:
    w_stat, p_val_w = 0, 1.0
print(f"T-test: t={t_stat:.3f}, p={p_val_t:.3e}")
print(f"Wilcoxon: W={w_stat}, p={p_val_w:.3e}")
