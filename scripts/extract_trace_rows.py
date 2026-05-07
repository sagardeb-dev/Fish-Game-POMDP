"""Extract per-trace event-log rows for Appendix F.

Given a panel events.jsonl and a key (e.g.
"panel=active|method=llm_raw|level=3|seed=362855691|model=gpt-5.5"),
emit a CSV with columns: step, action, args, obs_summary.

One row per llm_action event (args from the preceding llm_model_call's parsed_action,
obs_summary from the following llm_tool_result / llm_intervention_result), plus a
final row from work_success.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def fmt_args(parsed: dict) -> str:
    a = parsed.get("action")
    if a == "intervene":
        return f"var={parsed.get('var')}, val={parsed.get('value')}"
    if a == "correlation":
        return f"i={parsed.get('i')}, j={parsed.get('j')}"
    if a == "partial_correlation":
        cond = parsed.get("conditioning_on", [])
        return f"i={parsed.get('i')}, j={parsed.get('j')}, cond={{{','.join(map(str, cond))}}}"
    if a == "independence_test":
        cond = parsed.get("conditioning_on", [])
        return f"i={parsed.get('i')}, j={parsed.get('j')}, cond={{{','.join(map(str, cond))}}}"
    if a == "submit_graph":
        nd = len(parsed.get("directed_edges", []))
        nu = len(parsed.get("undirected_edges", []))
        return f"|E_d|={nd}, |E_u|={nu}"
    return ""


def fmt_obs(event_type: str, payload: dict) -> str:
    if event_type == "llm_intervention_result":
        return f"{payload.get('rows')} samples; budget left {payload.get('remaining_budget')}"
    if event_type == "llm_tool_result":
        v = payload.get("value")
        ind = payload.get("independent")
        if ind is not None:
            return f"indep={ind}, p={payload.get('p_value'):.3g}" if payload.get("p_value") is not None else f"indep={ind}"
        if v is not None:
            return f"value={v:.3f}" if isinstance(v, float) else f"value={v}"
        return ""
    if event_type == "work_success":
        f1 = payload.get("directed_f1")
        shd = payload.get("dag_shd")
        return f"dir_f1={f1:.3f}, SHD={shd}"
    return ""


def extract(events_path: Path, key: str, out_csv: Path) -> None:
    matched = []
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            if f'"key":"{key}"' not in line:
                continue
            matched.append(json.loads(line))

    if not matched:
        raise SystemExit(f"no events matched key: {key}")

    rows = []
    pending_args = ""
    pending_action = ""
    pending_step = None

    for ev in matched:
        et = ev["event_type"]
        p = ev.get("payload", {})
        if et == "llm_model_call":
            parsed = p.get("parsed_action", {})
            pending_action = parsed.get("action", "")
            pending_args = fmt_args(parsed)
            pending_step = p.get("step")
        elif et == "llm_action":
            rows.append({
                "step": p.get("step", pending_step),
                "action": pending_action,
                "args": pending_args,
                "obs_summary": "",
            })
        elif et in ("llm_intervention_result", "llm_tool_result"):
            if rows:
                rows[-1]["obs_summary"] = fmt_obs(et, p)
        elif et == "work_success":
            if rows and rows[-1]["action"] == "submit_graph":
                rows[-1]["obs_summary"] = fmt_obs(et, p)
            else:
                rows.append({
                    "step": (rows[-1]["step"] + 1) if rows else 1,
                    "action": "submit_graph",
                    "args": f"|E_d|={len(p.get('directed_edges', []))}, |E_u|={len(p.get('undirected_edges', []))}",
                    "obs_summary": fmt_obs(et, p),
                })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "action", "args", "obs_summary"])
        w.writeheader()
        w.writerows(rows)
    print(f"[done] {out_csv}  ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    extract(args.events, args.key, args.out)


if __name__ == "__main__":
    main()
