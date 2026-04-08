"""
Run 5 baselines on BENCHMARK_CONFIG (tightened distributions).

Runs: 5 baselines × 5 seeds × 1 ablation (full tools) = 25 episodes
Prints results table with reward, Brier scores, gaps.
Compares to predictions from baseline_prediction.py.

No modifications to simulator/config/baselines — just runs existing code.
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from fishing_game.config import CONFIG
from fishing_game.simulator import FishingGameEnv
from fishing_game.evaluator import Evaluator
from fishing_game.baselines import (
    RandomAgent, NaivePatternMatcher, CausalLearner,
    CausalReasoner, OracleAgent,
)

BASELINES = [
    ("Random", RandomAgent),
    ("NaivePattern", NaivePatternMatcher),
    ("CausalLearner", CausalLearner),
    ("CausalReasoner", CausalReasoner),
    ("Oracle", OracleAgent),
]

SEEDS = [42, 123, 456, 789, 1024]


def run_episode(agent_name, agent_cls, seed):
    """Run a single episode, return full evaluation dict."""
    env = FishingGameEnv(config=CONFIG)
    obs = env.reset(seed=seed)
    agent = agent_cls(config=CONFIG)
    if hasattr(agent, "reset"):
        agent.reset()

    total_reward = 0.0
    for _ in range(CONFIG["episode_length"]):
        result = agent.act(env, obs)
        total_reward += result["reward"]
        if result["done"]:
            break
        obs = result["observation"]

    trace = env.get_trace()
    evaluator = Evaluator(config=CONFIG)
    eval_result = evaluator.evaluate_episode(trace)

    return {
        "agent": agent_name,
        "seed": seed,
        "total_reward": eval_result.get("total_reward", total_reward),
        "mean_brier_storm": eval_result.get("mean_brier_storm", -1),
        "mean_brier_equip": eval_result.get("mean_brier_equip", -1),
        "total_tool_use_gap": eval_result.get("total_tool_use_gap", 0),
        "total_inference_gap": eval_result.get("total_inference_gap", 0),
        "total_planning_gap": eval_result.get("total_planning_gap", 0),
    }


def main():
    print("=" * 90)
    print("BASELINE ABLATION: 5 baselines × 5 seeds")
    print("CONFIG: BENCHMARK_CONFIG (sensor_zones=2, tightened distributions)")
    print("=" * 90)

    results_by_agent = {name: [] for name, _ in BASELINES}

    # Run all episodes in parallel
    tasks = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for agent_name, agent_cls in BASELINES:
            for seed in SEEDS:
                future = executor.submit(run_episode, agent_name, agent_cls, seed)
                tasks.append(future)

        completed = 0
        for future in as_completed(tasks):
            result = future.result()
            results_by_agent[result["agent"]].append(result)
            completed += 1
            print(f"[{completed}/{len(tasks)}] {result['agent']:16} seed={result['seed']:4} reward={result['total_reward']:7.1f}")

    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)

    # Aggregate stats
    print(f"\n{'Agent':<16} {'Mean':>8} {'Std':>7}  {'Brier(S)':>9} {'Brier(E)':>9}  {'ToolGap':>9} {'InfGap':>9}")
    print("-" * 90)

    for agent_name, agent_cls in BASELINES:
        evals = results_by_agent[agent_name]
        rewards = [e["total_reward"] for e in evals]
        brier_s = [e["mean_brier_storm"] for e in evals if e["mean_brier_storm"] >= 0]
        brier_e = [e["mean_brier_equip"] for e in evals if e["mean_brier_equip"] >= 0]
        tool_gaps = [e["total_tool_use_gap"] for e in evals]
        inf_gaps = [e["total_inference_gap"] for e in evals]

        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        mean_brier_s = np.mean(brier_s) if brier_s else 0.0
        mean_brier_e = np.mean(brier_e) if brier_e else 0.0
        mean_tool_gap = np.mean(tool_gaps)
        mean_inf_gap = np.mean(inf_gaps)

        print(
            f"{agent_name:<16} {mean_reward:>8.1f} {std_reward:>6.1f}  "
            f"{mean_brier_s:>9.4f} {mean_brier_e:>9.4f}  "
            f"{mean_tool_gap:>9.1f} {mean_inf_gap:>9.1f}"
        )

    # Per-seed table
    print(f"\n{'Agent':<16} {'Mean':>8} {'Std':>6}  Per-seed")
    print("-" * 70)
    for agent_name, agent_cls in BASELINES:
        evals = results_by_agent[agent_name]
        rewards = [e["total_reward"] for e in evals]
        per_seed = "  ".join(f"s{e['seed']}={e['total_reward']:.0f}" for e in evals)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        print(f"{agent_name:<16} {mean_reward:>8.1f} {std_reward:>5.1f}  {per_seed}")

    # Compare to predictions
    print("\n" + "=" * 90)
    print("PREDICTION vs ACTUAL")
    print("=" * 90)

    from baseline_prediction import predict_scores
    predictions = predict_scores()

    print(f"\n{'Agent':<16} {'Predicted':>12} {'Actual':>12} {'Delta':>10} {'Status':<15}")
    print("-" * 70)

    for agent_name, agent_cls in BASELINES:
        evals = results_by_agent[agent_name]
        rewards = [e["total_reward"] for e in evals]
        actual_mean = np.mean(rewards)

        pred_mean, pred_std = predictions[agent_name]
        delta = actual_mean - pred_mean

        # Status: within 1 std, close, or off
        if abs(delta) <= pred_std:
            status = "✓ ON TARGET"
        elif abs(delta) <= 2 * pred_std:
            status = "~ CLOSE"
        else:
            status = "✗ OFF"

        print(f"{agent_name:<16} {pred_mean:>6.0f}±{pred_std:<3.0f}     {actual_mean:>9.1f}     {delta:>9.1f}  {status}")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
