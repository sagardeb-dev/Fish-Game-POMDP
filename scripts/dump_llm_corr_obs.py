"""Reconstruct llm_corr_obs prompts + parsed actions to a root markdown file."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVENTS = ROOT / "traces/ladder/corr_obs_probe/events_corr_obs.jsonl"
OUT = ROOT / "llm_call_llm_corr_obs.md"

sys.path.insert(0, str(ROOT))
from run_corr_obs_probe import (  # noqa: E402
    build_corr_obs_system_prompt,
    build_corr_obs_session_prompt,
)

import numpy as np  # noqa: E402

PICKS = [
    ("panel=observational|method=llm_corr_obs|level=1|seed=1257633785|model=gpt-5.4"),
    ("panel=observational|method=llm_corr_obs|level=1|seed=1257633785|model=claude-sonnet-4-6"),
]


def load_episodes() -> dict[str, dict]:
    by_key: dict[str, dict] = {}
    for line in EVENTS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        key = ev.get("key")
        if key not in PICKS:
            continue
        ep = by_key.setdefault(key, {})
        ep[ev["event_type"]] = ev["payload"]
    return by_key


def render(key: str, ep: dict) -> str:
    summary = ep["corr_obs_summary"]
    submit = ep["corr_obs_submit"]
    success = ep.get("work_success", {})

    system = build_corr_obs_system_prompt()
    user = build_corr_obs_session_prompt(
        variable_names=tuple(summary["variables"]),
        n_obs=int(summary["n_obs"]),
        std=np.array(summary["std"]),
        corr=np.array(summary["corr"]),
    )
    user_msg = (
        f"Session data JSON: {user}\n"
        "Call the causal_discovery_action tool exactly once for your next action."
    )
    parts = [
        f"# {key}\n",
        "_Reconstructed: raw response body was not logged by run_corr_obs_probe.py; only parsed action survives._\n",
        "## System prompt\n",
        f"```\n{system}\n```\n",
        "## User message\n",
        f"```\n{user_msg}\n```\n",
        "## Parsed action\n",
        f"```json\n{json.dumps(submit, indent=2)}\n```\n",
        "## Score\n",
        f"```json\n{json.dumps(success, indent=2)}\n```\n",
    ]
    return "\n".join(parts)


def main() -> None:
    eps = load_episodes()
    blocks = []
    for key in PICKS:
        if key not in eps:
            continue
        blocks.append(render(key, eps[key]))
        blocks.append("\n---\n")
    OUT.write_text("\n".join(blocks), encoding="utf-8")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
