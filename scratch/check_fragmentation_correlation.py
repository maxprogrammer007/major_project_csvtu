import pandas as pd
from scipy.stats import pearsonr, spearmanr

df = pd.read_csv('d:/RRC/major_project_csvtu/splerge_h2_per_table_metrics.csv')

# `raw_cells` is the number of grid tiles the split model predicted
# `destroyed_tp_25` is the number of TPs destroyed by the heuristic
# We want to correlate fragmentation (raw_cells) with destruction (destroyed_tp_25)

print("Correlation between raw_cells and destroyed_tp_25:")
pearson_corr, p_val_p = pearsonr(df['raw_cells'], df['destroyed_tp_25'])
print(f"Pearson: r = {pearson_corr:.3f}, p = {p_val_p:.3e}")

spearman_corr, p_val_s = spearmanr(df['raw_cells'], df['destroyed_tp_25'])
print(f"Spearman: rho = {spearman_corr:.3f}, p = {p_val_s:.3e}")
