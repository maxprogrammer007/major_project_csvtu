import csv
import numpy as np
from scipy import stats

def main():
    table_ids = []
    raw_tps = []
    heur_tps = []
    h2_tps = []
    gts = []

    with open('splerge_h2_per_table_metrics.csv', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            table_ids.append(row['table_id'])
            raw_tps.append(int(row['raw_tp_25']))
            heur_tps.append(int(row['heur_tp_25']))
            h2_tps.append(int(row['h2_tp_25']))
            gts.append(int(row['gt_count']))

    raw_tps = np.array(raw_tps)
    heur_tps = np.array(heur_tps)
    h2_tps = np.array(h2_tps)
    gts = np.array(gts)

    heur_rec = heur_tps / gts
    h2_rec = h2_tps / gts
    diff_rec = h2_rec - heur_rec

    # Paired test
    diff_tp = h2_tps - heur_tps
    res_wilcoxon = stats.wilcoxon(diff_tp, alternative='greater')
    res_ttest = stats.ttest_rel(h2_tps, heur_tps, alternative='greater')

    # 95% CI on mean per-table recall difference
    mean_diff = np.mean(diff_rec)
    sem_diff = stats.sem(diff_rec)
    ci_low = mean_diff - 1.96 * sem_diff
    ci_high = mean_diff + 1.96 * sem_diff

    print("=== Statistical Analysis: H2 Set Packing vs SPLERGE Heuristic Merge (N=100) ===")
    print(f"Mean Per-Table Recall Difference: +{mean_diff*100:.2f}% [95% CI: +{ci_low*100:.2f}%, +{ci_high*100:.2f}%]")
    print(f"Paired t-test: t = {res_ttest.statistic:.3f}, p-value = {res_ttest.pvalue:.2e}")
    print(f"Wilcoxon signed-rank test: W = {res_wilcoxon.statistic:.1f}, p-value = {res_wilcoxon.pvalue:.2e}")

    # Win / Tie / Loss
    wins = np.sum(diff_tp > 0)
    ties = np.sum(diff_tp == 0)
    losses = np.sum(diff_tp < 0)
    print(f"Per-Table Win/Tie/Loss: Wins={wins}, Ties={ties}, Losses={losses}")

if __name__ == "__main__":
    main()
