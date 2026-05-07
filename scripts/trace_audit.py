"""Quick audit of LLM traces from a smoke run."""
import json, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

EVENTS_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"C:\projects\Random Research\Internet of Agents Benchmark\RL-environment"
    r"\traces\ladder\l3_v1_all_policy_smoke_deepseek_v32\events.jsonl"
)

events = []
for line in EVENTS_FILE.read_text(encoding="utf-8").splitlines():
    if line.strip():
        events.append(json.loads(line))

policies = sorted(set(ev["key"] for ev in events if ev["event_type"] == "llm_model_call"))

for policy_key in policies:
    method = policy_key.split("|")[1].split("=")[1]
    print(f"\n{'='*80}")
    print(f"POLICY: {method}")
    print(f"{'='*80}")

    model_calls = [ev for ev in events if ev["event_type"] == "llm_model_call" and ev["key"] == policy_key]
    tool_results = [ev for ev in events if ev["event_type"] == "llm_tool_result" and ev["key"] == policy_key]
    interventions = [ev for ev in events if ev["event_type"] == "llm_intervention_result" and ev["key"] == policy_key]

    print(f"LLM calls: {len(model_calls)}, tool results: {len(tool_results)}, interventions: {len(interventions)}")

    for mc in model_calls:
        p = mc["payload"]
        pa = p.get("parsed_action") or {}
        step = p.get("step", "?")
        status = p.get("status", "ok")
        error = p.get("error")
        if not pa:
            print(f"  Step {step:>2}: PARSE_FAIL status={status} error={str(error)[:120] if error else 'none'}")
            continue
        act = pa.get("action", "?")

        if act == "submit_graph":
            de = pa.get("directed_edges", [])
            ue = pa.get("undirected_edges", [])
            rs = pa.get("reasoning_summary", "")[:200]
            print(f"  Step {step:>2}: SUBMIT  directed={len(de)} undirected={len(ue)}")
            print(f"          reasoning: {rs}")
        elif act in ("correlation", "partial_correlation", "independence_test"):
            i = pa.get("i", "")
            j = pa.get("j", "")
            cond = pa.get("conditioning_on", [])
            alpha = pa.get("alpha")
            print(f"  Step {step:>2}: {act:<22} ({i},{j}) cond={cond}" + (f" alpha={alpha}" if alpha else ""))
        elif act == "intervene":
            var = pa.get("var", pa.get("variable", "?"))
            val = pa.get("value", "?")
            rs = pa.get("reasoning_summary", "")[:100]
            print(f"  Step {step:>2}: INTERVENE var={var} val={val}  | {rs}")
        else:
            print(f"  Step {step:>2}: {act} {json.dumps(pa)[:120]}")

    work_success = [ev for ev in events if ev["event_type"] == "work_success" and ev["key"] == policy_key]
    if work_success:
        ws = work_success[0]["payload"]

        def _get_f1(val):
            if isinstance(val, dict):
                return val.get("f1", "?")
            return val

        sf1 = _get_f1(ws.get("skeleton_f1", "?"))
        df1 = _get_f1(ws.get("directed_f1", "?"))
        shd = ws.get("dag_shd", "?")
        eff = ws.get("efficiency", "?")
        sf1_s = f"{sf1:.3f}" if isinstance(sf1, (int, float)) else str(sf1)
        df1_s = f"{df1:.3f}" if isinstance(df1, (int, float)) else str(df1)
        eff_s = f"{eff:.3f}" if isinstance(eff, (int, float)) else str(eff)
        print(f"  SCORE: skel_f1={sf1_s} dir_f1={df1_s} shd={shd} eff={eff_s}")

print(f"\n{'='*80}")
print("SYSTEM PROMPTS (first call per policy)")
print(f"{'='*80}")
for policy_key in policies:
    method = policy_key.split("|")[1].split("=")[1]
    first_call = next((ev for ev in events if ev["event_type"] == "llm_model_call" and ev["key"] == policy_key), None)
    if first_call:
        msgs = first_call["payload"]["request"]["messages"]
        sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "NO SYSTEM")
        print(f"\n--- {method} ---")
        print(sys_msg)
        print()
