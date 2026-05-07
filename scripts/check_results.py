import csv
with open(r"C:\projects\Random Research\Internet of Agents Benchmark\RL-environment\traces\ladder\l3_v1_all_policy_smoke_deepseek_v32\results_long.csv", newline="") as f:
    rows = list(csv.DictReader(f))
for r in rows:
    panel = r["panel"]
    method = r["method"]
    seed = r["seed"]
    status = r["status"]
    print(f"{panel:<15} {method:<28} seed={seed}  status={status}")
