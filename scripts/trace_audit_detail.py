"""Detailed trace audit: final submissions, intervention results, tool results."""
import json, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

EVENTS_FILE = Path(
    r"C:\projects\Random Research\Internet of Agents Benchmark\RL-environment"
    r"\traces\ladder\l3_v1_all_policy_smoke_deepseek_v32\events.jsonl"
)

events = []
for line in EVENTS_FILE.read_text(encoding="utf-8").splitlines():
    if line.strip():
        events.append(json.loads(line))

# Final submissions for each policy
print("=" * 80)
print("FINAL SUBMISSIONS")
print("=" * 80)
policies = sorted(set(ev["key"] for ev in events if ev["event_type"] == "llm_model_call"))
for key in policies:
    method = key.split("|")[1].split("=")[1]
    calls = [ev for ev in events if ev["event_type"] == "llm_model_call" and ev["key"] == key]
    for c in calls:
        pa = c["payload"].get("parsed_action", {})
        if pa.get("action") == "submit_graph":
            de = pa.get("directed_edges", [])
            ue = pa.get("undirected_edges", [])
            print(f"\n--- {method} ---")
            print(f"  directed_edges ({len(de)}): {de}")
            print(f"  undirected_edges ({len(ue)}): {ue}")
            rs = pa.get("reasoning_summary", "")
            print(f"  reasoning: {rs[:400]}")

# Intervention results
print("\n" + "=" * 80)
print("INTERVENTION RESULTS")
print("=" * 80)
for ev in events:
    if ev["event_type"] == "llm_intervention_result":
        method = ev["key"].split("|")[1].split("=")[1]
        p = ev["payload"]
        data = p.get("intervention_data", [])
        print(f"\n--- {method} ---")
        print(f"  variable={p.get('variable')} value={p.get('value')} samples={len(data)}")
        if data:
            print(f"  first row (truncated): {str(data[0])[:150]}")

# Tool result samples for llm_stats_obs
print("\n" + "=" * 80)
print("TOOL RESULT SAMPLES (llm_stats_obs)")
print("=" * 80)
tool_results = [ev for ev in events if ev["event_type"] == "llm_tool_result" and "llm_stats_obs" in ev["key"]]
for tr in tool_results[:10]:
    p = tr["payload"]
    print(f"  Step {p.get('step')}: {p.get('tool_name','?')} -> {json.dumps(p.get('result'))[:120]}")

# Check for conditioning_on=-5 issue
print("\n" + "=" * 80)
print("ANOMALOUS ACTIONS (negative indices, repeated queries)")
print("=" * 80)
for ev in events:
    if ev["event_type"] == "llm_model_call":
        pa = ev["payload"].get("parsed_action", {})
        cond = pa.get("conditioning_on", [])
        if cond and any(c < 0 for c in cond):
            method = ev["key"].split("|")[1].split("=")[1]
            step = ev["payload"]["step"]
            print(f"  {method} step {step}: negative conditioning index! cond={cond}")

# Instance metadata (true DAG info)
print("\n" + "=" * 80)
print("INSTANCE METADATA (first key)")
print("=" * 80)
meta = [ev for ev in events if ev["event_type"] == "instance_metadata"]
if meta:
    p = meta[0]["payload"]
    print(f"  d={p.get('d')} k={p.get('k')} budget={p.get('intervention_budget')}")
    print(f"  true_edges: {p.get('true_edges')}")
    print(f"  optimal_intervention_set: {p.get('optimal_intervention_set')}")
    cpdag = p.get("observational_ceiling", {})
    print(f"  cpdag directed: {cpdag.get('directed_edges')}")
    print(f"  cpdag undirected: {cpdag.get('undirected_edges')}")
