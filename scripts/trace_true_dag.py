"""Extract true DAG and compare against each policy's submission."""
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

meta_evs = [ev for ev in events if ev["event_type"] == "instance_metadata"]
print("Instance metadata keys:", list(meta_evs[0]["payload"].keys()))
print()

# Get true DAG from first instance_metadata
p = meta_evs[0]["payload"]
true_edges = set()
for key in p:
    if "true" in key.lower() and "edge" in key.lower():
        val = p[key]
        if isinstance(val, list):
            true_edges = {tuple(e) for e in val}
            print(f"True DAG edges (from '{key}'): {sorted(true_edges)}")
            break

if not true_edges:
    print("Could not find true edges. Full payload:")
    print(json.dumps(p, indent=2)[:2000])

cpdag_dir = set()
cpdag_undir = set()
for key in p:
    if "cpdag" in key.lower() and "directed" in key.lower() and "un" not in key.lower():
        cpdag_dir = {tuple(e) for e in p[key]} if p[key] else set()
    if "cpdag" in key.lower() and "undirected" in key.lower():
        cpdag_undir = {tuple(e) for e in p[key]} if p[key] else set()

print(f"CPDAG directed: {sorted(cpdag_dir)}")
print(f"CPDAG undirected: {sorted(cpdag_undir)}")
print(f"Optimal intervention set: {p.get('optimal_intervention_set')}")
print(f"Budget: {p.get('intervention_budget')}")
config = p.get("config", {})
print(f"d={config.get('d')} k={config.get('k')}")
print()

# Compare each policy's final submission
print("=" * 80)
print("SUBMISSION vs TRUE DAG")
print("=" * 80)

policies = sorted(set(ev["key"] for ev in events if ev["event_type"] == "llm_model_call"))
for key in policies:
    method = key.split("|")[1].split("=")[1]
    seed = key.split("|")[3].split("=")[1]
    calls = [ev for ev in events if ev["event_type"] == "llm_model_call" and ev["key"] == key]
    submissions = [c for c in calls if c["payload"].get("parsed_action", {}).get("action") == "submit_graph"]
    if not submissions:
        continue
    last = submissions[-1]
    pa = last["payload"]["parsed_action"]
    sub_dir = {tuple(e) for e in pa.get("directed_edges", [])}
    sub_undir = {tuple(e) for e in pa.get("undirected_edges", [])}

    # Compute edge-level accuracy
    correct_dir = sub_dir & true_edges
    reversed_dir = {(b, a) for (a, b) in sub_dir} & true_edges
    wrong_dir = sub_dir - true_edges - {(b, a) for (a, b) in true_edges}
    true_skel = {frozenset(e) for e in true_edges}
    sub_skel = {frozenset(e) for e in sub_dir} | {frozenset(e) for e in sub_undir}
    skel_correct = true_skel & sub_skel
    skel_extra = sub_skel - true_skel
    skel_missing = true_skel - sub_skel

    print(f"\n--- {method} (seed={seed}) ---")
    print(f"  Submitted: {len(sub_dir)} directed, {len(sub_undir)} undirected")
    print(f"  Correct direction:  {sorted(correct_dir)}")
    print(f"  Reversed direction: {sorted(reversed_dir)}")
    print(f"  Spurious edges:     {sorted(wrong_dir)}")
    print(f"  Skeleton correct:   {len(skel_correct)}/{len(true_skel)}")
    print(f"  Skeleton extra:     {sorted(skel_extra)}")
    print(f"  Skeleton missing:   {len(skel_missing)}/{len(true_skel)}")
