"""Run baselines and GPT agent on HARD_CONFIG, print comparison."""

import random as stdlib_random
import numpy as np
from fishing_game.config import CONFIG, HARD_CONFIG
from fishing_game.simulator import FishingGameEnv
from fishing_game.evaluator import Evaluator
from fishing_game.baselines import (
    RandomAgent, NoToolsHeuristic, SearchOnlyHeuristic,
    BeliefAwareBaseline, OracleAgent,
)

BASELINES = [
    ("Random", RandomAgent),
    ("NoTools", NoToolsHeuristic),
    ("SearchOnly", SearchOnlyHeuristic),
    ("BeliefAware", BeliefAwareBaseline),
    ("Oracle", OracleAgent),
]

SEEDS = [42, 123, 456, 789, 1024]


def run_episode(agent_cls, seed, config):
    env = FishingGameEnv(config=config)
    obs = env.reset(seed=seed)
    agent = agent_cls(config=config)
    if hasattr(agent, "reset"):
        agent.reset()

    rng = stdlib_random.Random(seed + 1000)
    total_reward = 0.0

    for _ in range(config["episode_length"]):
        result = agent.act(env, obs, rng=rng)
        total_reward += result["reward"]
        if result["done"]:
            break
        obs = result["observation"]

    trace = env.get_trace()
    evaluator = Evaluator(config=config)
    eval_result = evaluator.evaluate_episode(trace)
    return total_reward, eval_result


def run_baselines(config, label):
    print(f"\n{'='*95}")
    print(f"  {label}")
    print(f"{'='*95}")

    header = (
        f"{'Agent':<14} {'Reward':>12} "
        f"{'Brier(S)':>9} {'Brier(Z)':>9} "
        f"{'ToolGap':>9} {'InfGap':>9} {'PlanGap':>9}"
    )
    print(header)
    print("-" * len(header))

    results = {}
    for name, cls in BASELINES:
        rewards = []
        brier_s = []
        brier_z = []
        tool_gaps = []
        inf_gaps = []
        plan_gaps = []

        for seed in SEEDS:
            total, ev = run_episode(cls, seed, config)
            rewards.append(total)
            brier_s.append(ev["mean_brier_storm"])
            brier_z.append(ev["mean_brier_zone"])
            tool_gaps.append(ev["total_tool_use_gap"])
            inf_gaps.append(ev["total_inference_gap"])
            plan_gaps.append(ev["total_planning_gap"])

        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        results[name] = mean_r

        print(
            f"{name:<14} {mean_r:>7.1f}+-{std_r:>4.1f} "
            f"{np.mean(brier_s):>9.4f} {np.mean(brier_z):>9.4f} "
            f"{np.mean(tool_gaps):>9.1f} {np.mean(inf_gaps):>9.1f} "
            f"{np.mean(plan_gaps):>9.1f}"
        )

    return results


def run_gpt_hard():
    """Run GPT agent on HARD_CONFIG."""
    try:
        from fishing_game.gpt_agent import GPTAgent
        from fishing_game.traced_runner import run_traced_episode, print_traced_episode
    except ImportError as e:
        print(f"\nSkipping GPT run: {e}")
        return None

    print(f"\n{'='*95}")
    print("  GPT-5.4 on HARD_CONFIG (seed=42)")
    print(f"{'='*95}")

    output = run_traced_episode(
        agent_cls=GPTAgent,
        seed=42,
        config=HARD_CONFIG,
        save_path="traces/gpt54_hard_seed42.json",
    )

    ev = output["evaluation"]
    print(f"  Total reward:       {ev['total_reward']}")
    print(f"  Brier (storm):      {ev['mean_brier_storm']:.4f}")
    print(f"  Brier (zone):       {ev['mean_brier_zone']:.4f}")
    print(f"  Tool use gap:       {ev['total_tool_use_gap']:.1f}")
    print(f"  Inference gap:      {ev['total_inference_gap']:.1f}")
    print(f"  Planning gap:       {ev['total_planning_gap']:.1f}")
    print(f"  Tool usage:         {ev['tool_usage_counts']}")
    print(f"  Reward/quarter:     {ev['reward_per_quarter']}")
    print(f"\n  Trace saved to traces/gpt54_hard_seed42.json")

    return ev["total_reward"]


if __name__ == "__main__":
    # Compare easy vs hard baselines
    easy = run_baselines(CONFIG, "EASY MODE (default CONFIG)")
    hard = run_baselines(HARD_CONFIG, "HARD MODE (HARD_CONFIG)")

    # Show the gap
    print(f"\n{'='*95}")
    print("  EASY vs HARD comparison")
    print(f"{'='*95}")
    print(f"{'Agent':<14} {'Easy':>10} {'Hard':>10} {'Drop':>10}")
    print("-" * 50)
    for name in easy:
        drop = easy[name] - hard[name]
        print(f"{name:<14} {easy[name]:>10.1f} {hard[name]:>10.1f} {drop:>+10.1f}")

    # Run GPT on hard mode
    gpt_reward = run_gpt_hard()
