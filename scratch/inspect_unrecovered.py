import csv

with open('splerge_h2_per_table_metrics.csv', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        dest = int(row['destroyed_tp_25'])
        rec = int(row['recovered_tp_25'])
        if dest != rec:
            print(f"Table {row['table_id']}: destroyed={dest}, recovered={rec}, raw_tp={row['raw_tp_25']}, heur_tp={row['heur_tp_25']}, h2_tp={row['h2_tp_25']}")
