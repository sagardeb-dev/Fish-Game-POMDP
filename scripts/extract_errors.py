"""Extract all error details from events."""
import json, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Check both v4flash dirs
for d in ["l3_v1_smoke_v4flash_clean", "l3_v1_smoke_deepseek_v4_flash"]:
    p = Path(r"C:\projects\Random Research\Internet of Agents Benchmark\RL-environment\traces\ladder") / d / "events.jsonl"
    if not p.exists():
        continue
    print(f"=== {d} ===")
    events = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    for ev in events:
        if ev["event_type"] == "llm_model_call":
            payload = ev["payload"]
            if payload.get("status") == "failed" or payload.get("error"):
                method = ev["key"].split("|")[1].split("=")[1]
                step = payload.get("step")
                error = payload.get("error", "")
                print(f"\n  {method} step {step}:")
                print(f"  error: {error[:500]}")
                # Check raw_response too
                raw = payload.get("raw_response")
                if raw:
                    print(f"  raw_response: {json.dumps(raw)[:300]}")
    print()
