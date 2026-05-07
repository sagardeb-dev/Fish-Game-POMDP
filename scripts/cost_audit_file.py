"""Cost audit for any events.jsonl file."""
import json, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

EVENTS_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"C:\projects\Random Research\Internet of Agents Benchmark\RL-environment"
    r"\traces\ladder\l3_v1_smoke_deepseek_v4_flash\events.jsonl"
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
        entry = by_method.setdefault(method, {"cost": 0.0, "calls": 0, "failed": 0})
        entry["cost"] += cost
        entry["calls"] += 1
        if ev["payload"].get("status") == "failed":
            entry["failed"] += 1

print(f"Events file: {EVENTS_FILE.name}")
print(f"Total LLM calls: {sum(d['calls'] for d in by_method.values())}")
print()
print(f"  {'Policy':<28} {'Calls':>5} {'Fail':>4}  {'Cost':>10}")
print(f"  {'-'*28} {'-'*5} {'-'*4}  {'-'*10}")
for m, d in sorted(by_method.items(), key=lambda x: -x[1]["cost"]):
    print(f"  {m:<28} {d['calls']:>5} {d['failed']:>4}  ${d['cost']:.4f}")
print()
print(f"  Total cost: ${total_cost:.4f}")
print(f"  Failed calls: {sum(d['failed'] for d in by_method.values())}")
