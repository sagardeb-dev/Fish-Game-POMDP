"""Check user messages fed to LLM on intervention steps."""
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

# llm_raw step 2 and 3 user messages (after first intervention)
print("=" * 80)
print("llm_raw STEP 2 user message (after first intervention)")
print("=" * 80)
for ev in events:
    if ev["event_type"] == "llm_model_call" and "method=llm_raw|" in ev["key"]:
        p = ev["payload"]
        if p["step"] == 2:
            msgs = p["request"]["messages"]
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            # Show first 2000 chars
            print(user_msg[:2000])
            print("...")
            # Check if intervention data present
            if "intervention" in user_msg.lower():
                print("\n[INTERVENTION DATA PRESENT IN USER MSG]")
            else:
                print("\n[NO INTERVENTION DATA IN USER MSG]")

print("\n" + "=" * 80)
print("llm_raw STEP 3 user message (final, after 2nd intervention)")
print("=" * 80)
for ev in events:
    if ev["event_type"] == "llm_model_call" and "method=llm_raw|" in ev["key"]:
        p = ev["payload"]
        if p["step"] == 3:
            msgs = p["request"]["messages"]
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            print(user_msg[:2000])
            print("...")

# pc_cpdag_llm step 1 user message - check what PC graph it receives
print("\n" + "=" * 80)
print("pc_cpdag_llm STEP 1 user message (what PC graph it gets)")
print("=" * 80)
for ev in events:
    if ev["event_type"] == "llm_model_call" and "method=pc_cpdag_llm|" in ev["key"]:
        p = ev["payload"]
        if p["step"] == 1:
            msgs = p["request"]["messages"]
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            print(user_msg[:3000])
            print("...")

# llm_stats step 40 tool schema - confirm it's forced to submit_graph
print("\n" + "=" * 80)
print("llm_stats STEP 40 tool schema enum")
print("=" * 80)
for ev in events:
    if ev["event_type"] == "llm_model_call" and "method=llm_stats|" in ev["key"]:
        p = ev["payload"]
        if p["step"] == 40:
            tools = p["request"].get("tools", [])
            if tools:
                params = tools[0].get("function", {}).get("parameters", {})
                action_enum = params.get("properties", {}).get("action", {}).get("enum", [])
                print(f"  action enum: {action_enum}")
