"""Cost audit for a smoke run."""
import json, sys
from pathlib import Path

EVENTS_FILE = Path(
    r"C:\projects\Random Research\Internet of Agents Benchmark\RL-environment"
    r"\traces\ladder\l3_v1_all_policy_smoke_deepseek_v32\events.jsonl"
)

events = []
for line in EVENTS_FILE.read_text(encoding="utf-8").splitlines():
    if line.strip():
        events.append(json.loads(line))

total_cost = 0.0
by_method: dict[str, dict] = {}
for ev in events:
    if ev["event_type"] == "llm_model_call":
        method = ev["key"].split("|")[1].split("=")[1]
        cost = ev["payload"].get("cost_usd") or 0
        total_cost += cost
        entry = by_method.setdefault(method, {"cost": 0.0, "calls": 0})
        entry["cost"] += cost
        entry["calls"] += 1

print("Cost per policy (1 seed, L3, deepseek-v3.2):")
print(f"  {'Policy':<28} {'Calls':>5}  {'Cost':>10}")
print(f"  {'-'*28} {'-'*5}  {'-'*10}")
for m, d in sorted(by_method.items(), key=lambda x: -x[1]["cost"]):
    print(f"  {m:<28} {d['calls']:>5}  ${d['cost']:.4f}")
print()
print(f"  Total: ${total_cost:.4f}")

# Obs-only cost (from this same run, before deprecation took effect)
obs_cost = sum(
    d["cost"] for m, d in by_method.items() if m in {"llm_raw_obs", "llm_stats_obs"}
)
active_cost = total_cost - obs_cost

print()
print(f"Active policies only (post-deprecation): ${active_cost:.4f}")
if obs_cost > 0:
    print(f"Deprecated obs-only policies: ${obs_cost:.4f}")

print()
print("Projection to full ladder (6 levels x 8 seeds):")
print(f"  Old (with obs-only):    ${total_cost * 8 * 6:.2f}")
print(f"  New (without obs-only): ${active_cost * 8 * 6:.2f}")
print(f"  Saved:                  ${obs_cost * 8 * 6:.2f}")
