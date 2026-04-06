"""Run the Coding Agent on the Fishing Game."""

import json
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from fishing_game.config import CONFIG
from fishing_game.simulator import FishingGameEnv
from fishing_game.evaluator import Evaluator
from fishing_game.coding_agent import CodingAgent

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _normalize_trace_for_evaluator(trace):
    normalized = []
    for step in trace:
        step_copy = dict(step)
        for key in ("observations", "available_observations"):
            if key in step_copy and isinstance(step_copy[key], list):
                step_copy[key] = [tuple(item) if isinstance(item, list) else item for item in step_copy[key]]
        normalized.append(step_copy)
    return normalized


def _beliefs_complete(step):
    beliefs = step.get("beliefs", {})
    required = [
        "storm_active",
        "storm_zone_probs",
        "equip_failure_active",
        "equip_zone_probs",
        "tide_high",
    ]
    if any(field not in beliefs for field in required):
        return False
    return all(zone in beliefs["storm_zone_probs"] for zone in ("A", "B", "C", "D")) and all(
        zone in beliefs["equip_zone_probs"] for zone in ("A", "B", "C", "D")
    )


def run_episode(seed=42, model="gpt-5.4"):
    cfg = CONFIG
    env = FishingGameEnv(config=cfg)
    obs = env.reset(seed=seed)

    agent = CodingAgent(model=model, config=cfg)
    agent.reset()

    print(f"Running CodingAgent ({model}) seed={seed}")
    print("=" * 60)

    total_reward = 0.0
    for day in range(cfg["episode_length"]):
        t0 = time.time()
        result = agent.act(env, obs)
        elapsed = time.time() - t0

        total_reward += result["reward"]
        print(f"\n--- Day {day+1}: reward={result['reward']:.0f} "
              f"cumulative={total_reward:.0f} ({elapsed:.1f}s) ---\n")

        if result["done"]:
            break
        obs = result["observation"]

    # Save trace and evaluate
    trace = env.get_trace()
    model_slug = model.replace(".", "")
    save_path = Path(f"traces/coding_agent_{model_slug}_seed{seed}.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(trace, f, indent=2, default=str)
    print(f"Trace saved to {save_path}")

    evaluator = Evaluator(config=cfg)
    ev = evaluator.evaluate_episode(_normalize_trace_for_evaluator(trace))
    tool_counts = Counter()
    for step in trace:
        tool_counts.update(call["tool_name"] for call in step.get("tools_used", []))
    full_belief_days = sum(1 for step in trace if _beliefs_complete(step))

    print(f"\n{'='*60}")
    print(f"EPISODE SUMMARY (seed={seed})")
    print(f"{'='*60}")
    print(f"  Total reward:   {ev['total_reward']}")
    print(f"  Brier (storm):  {ev['mean_brier_storm']:.4f}")
    print(f"  Brier (equip):  {ev['mean_brier_equip']:.4f}")
    print(f"  Tool use gap:   {ev['total_tool_use_gap']:.1f}")
    print(f"  Inference gap:  {ev['total_inference_gap']:.1f}")
    print(f"  Planning gap:   {ev['total_planning_gap']:.1f}")
    print(f"  analyze_data:   {tool_counts.get('analyze_data', 0)} calls")
    print(f"  Full beliefs:   {full_belief_days}/{len(trace)} days")

    return ev


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-5.4"
    run_episode(seed=seed, model=model)
