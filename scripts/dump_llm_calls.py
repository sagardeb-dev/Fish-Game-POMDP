"""Dump LLM call (request + raw response) per policy episode to root markdown files."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVENTS = ROOT / "traces/ladder/deepseek_v4_flash_ladder_2seed/events.jsonl"

EPISODES = {
    "llm_call_llm_raw_obs.md": "panel=observational|method=llm_raw_obs|level=0|seed=1276453582|model=openrouter/deepseek/deepseek-v4-flash",
    "llm_call_llm_stats_obs.md": "panel=observational|method=llm_stats_obs|level=0|seed=1276453582|model=openrouter/deepseek/deepseek-v4-flash",
    "llm_call_llm_raw.md": "panel=active|method=llm_raw|level=0|seed=27998312|model=openrouter/deepseek/deepseek-v4-flash",
    "llm_call_llm_stats.md": "panel=active|method=llm_stats|level=0|seed=27998312|model=openrouter/deepseek/deepseek-v4-flash",
}


def render_step(payload: dict) -> str:
    request = payload.get("request", {})
    raw = payload.get("raw_response", {})
    messages = request.get("messages", [])
    system = next((m["content"] for m in messages if m.get("role") == "system"), "")
    user = next((m["content"] for m in messages if m.get("role") == "user"), "")
    tools = request.get("tools", [])
    parsed = payload.get("parsed_action")

    parts = [f"## Step {payload.get('step')}\n"]
    parts.append("### System prompt\n")
    parts.append(f"```\n{system}\n```\n")
    parts.append("### User message\n")
    parts.append(f"```\n{user}\n```\n")
    parts.append("### Tool schema\n")
    parts.append(f"```json\n{json.dumps(tools, indent=2)}\n```\n")
    parts.append("### Raw response\n")
    parts.append(f"```json\n{json.dumps(raw, indent=2)}\n```\n")
    parts.append("### Parsed action\n")
    parts.append(f"```json\n{json.dumps(parsed, indent=2)}\n```\n")
    parts.append(
        f"_status={payload.get('status')} latency={payload.get('latency_sec')}s "
        f"cost_usd={payload.get('cost_usd')} usage={json.dumps(payload.get('usage'))}_\n"
    )
    return "\n".join(parts)


def dump():
    by_key: dict[str, list[dict]] = {}
    for line in EVENTS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev.get("event_type") != "llm_model_call":
            continue
        by_key.setdefault(ev["key"], []).append(ev["payload"])

    for filename, key in EPISODES.items():
        steps = sorted(by_key.get(key, []), key=lambda p: p.get("step", 0))
        md = [f"# {key}\n", f"_Steps: {len(steps)}_\n"]
        for s in steps:
            md.append(render_step(s))
            md.append("\n---\n")
        (ROOT / filename).write_text("\n".join(md), encoding="utf-8")
        print(f"wrote {filename} ({len(steps)} steps)")


if __name__ == "__main__":
    dump()
