"""Deep audit: raw intervention payload, tool result payload, instance metadata."""
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

# Raw intervention result payloads
print("=" * 80)
print("RAW INTERVENTION RESULT PAYLOADS")
print("=" * 80)
for ev in events:
    if ev["event_type"] == "llm_intervention_result":
        method = ev["key"].split("|")[1].split("=")[1]
        print(f"\n--- {method} ---")
        print(json.dumps(ev["payload"], indent=2)[:600])

# Raw tool result payloads (first 3 per policy)
print("\n" + "=" * 80)
print("RAW TOOL RESULT PAYLOADS (first 3 per policy)")
print("=" * 80)
policies = set()
for ev in events:
    if ev["event_type"] == "llm_tool_result":
        method = ev["key"].split("|")[1].split("=")[1]
        if method not in policies:
            policies.add(method)
            print(f"\n--- {method} ---")
            same = [e for e in events if e["event_type"] == "llm_tool_result" and e["key"] == ev["key"]]
            for s in same[:3]:
                print(json.dumps(s["payload"], indent=2)[:400])
                print()

# Full instance_metadata payload
print("\n" + "=" * 80)
print("FULL INSTANCE METADATA (first)")
print("=" * 80)
meta = [ev for ev in events if ev["event_type"] == "instance_metadata"]
if meta:
    print(json.dumps(meta[0]["payload"], indent=2)[:2000])

# Working memory fed to llm_stats step 40 (final call)
print("\n" + "=" * 80)
print("WORKING MEMORY on llm_stats final step")
print("=" * 80)
for ev in events:
    if ev["event_type"] == "llm_model_call" and "method=llm_stats|" in ev["key"]:
        p = ev["payload"]
        if p.get("step") == 40:
            msgs = p["request"]["messages"]
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            # Extract working memory portion
            if "Working memory" in user_msg:
                idx = user_msg.index("Working memory")
                print(user_msg[idx:idx+1000])
            elif "working memory" in user_msg.lower():
                idx = user_msg.lower().index("working memory")
                print(user_msg[idx:idx+1000])
            else:
                print("No working memory found in user message")
                print("User message (last 500 chars):", user_msg[-500:])
