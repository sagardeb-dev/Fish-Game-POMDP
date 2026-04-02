"""
Ablation runner for the Fishing Game v5 — Discoverable causal structure.

3 ablation configs x 5 baselines x 10 seeds = 150 episodes.
Prints comparison table. Verifies baseline ordering, decomposition identity,
and tool_use_gap behavior (positive for non-SQL agents, ~0 for SQL agents).
"""

import os
import random as stdlib_random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from fishing_game.config import CONFIG
from fishing_game.simulator import FishingGameEnv
from fishing_game.evaluator import Evaluator
from fishing_game.baselines import (
    RandomAgent, NaivePatternMatcher, CausalLearner,
    CausalReasoner, OracleAgent,
)


# --- Ablation configurations ---
# In v3, raw sensors are always free. These only control budget-gated tools.
ABLATION_CONFIGS = {
    "full": {
        "check_weather_reports": True, "check_equipment_reports": True,
        "query_fishing_log": True, "query_maintenance_log": True,
        "analyze_data": True, "evaluate_options": True, "forecast_scenario": True,
    },
    "no_search": {
        "check_weather_reports": False, "check_equipment_reports": False,
        "query_fishing_log": False, "query_maintenance_log": False,
        "analyze_data": True, "evaluate_options": True, "forecast_scenario": True,
    },
    "no_tools": {
        "check_weather_reports": False, "check_equipment_reports": False,
        "query_fishing_log": False, "query_maintenance_log": False,
        "analyze_data": False, "evaluate_options": False, "forecast_scenario": False,
    },
}

BASELINES = [
    ("Random", RandomAgent),
    ("NaivePattern", NaivePatternMatcher),
    ("CausalLearner", CausalLearner),
    ("CausalReasoner", CausalReasoner),
    ("Oracle", OracleAgent),
]

DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 2048, 3000, 4096, 5555, 7777]


def run_episode(agent_cls, seed, config=None, ablation=None):
    """Run a single episode. Returns (total_reward, trace, eval_result)."""
    cfg = config or CONFIG
    env = FishingGameEnv(config=cfg, ablation=ablation)
    obs = env.reset(seed=seed)
    agent = agent_cls(config=cfg)
    if hasattr(agent, "reset"):
        agent.reset()

    rng = stdlib_random.Random(seed + 1000)
    total_reward = 0.0

    for _ in range(cfg["episode_length"]):
        result = agent.act(env, obs, rng=rng)
        total_reward += result["reward"]
        if result["done"]:
            break
        obs = result["observation"]

    trace = env.get_trace()
    evaluator = Evaluator(config=cfg)
    eval_result = evaluator.evaluate_episode(trace)

    return total_reward, trace, eval_result


def _run_episode_task(task):
    """Top-level wrapper for ProcessPoolExecutor (must be picklable)."""
    _, trace, eval_result = run_episode(
        task["agent_cls"], task["seed"],
        config=task["config"], ablation=task["ablation"],
    )
    return {
        "config_name": task["config_name"],
        "agent_name": task["agent_name"],
        "seed": task["seed"],
        "eval_result": eval_result,
    }


def run_ablation_suite(seeds=None, config=None, verify=True, max_workers=None):
    """Run all baselines x all ablation configs x all seeds.

    Args:
        max_workers: Number of parallel processes. Defaults to CPU count.
                     Set to 1 to disable parallelism (useful for debugging).
    """
    import numpy as np

    seeds = seeds or DEFAULT_SEEDS
    cfg = config or CONFIG
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    # Build task list
    tasks = []
    for config_name, ablation in ABLATION_CONFIGS.items():
        for agent_name, agent_cls in BASELINES:
            for seed in seeds:
                tasks.append({
                    "config_name": config_name,
                    "agent_name": agent_name,
                    "agent_cls": agent_cls,
                    "seed": seed,
                    "config": cfg,
                    "ablation": ablation,
                })

    total = len(tasks)
    completed = 0
    all_pass = True
    collected = {}

    if max_workers == 1:
        for task in tasks:
            result = _run_episode_task(task)
            completed += 1
            print(
                f"  [{completed:>3}/{total}] "
                f"{result['config_name']:<14} {result['agent_name']:<16} "
                f"seed={result['seed']:<5} "
                f"reward={result['eval_result']['total_reward']:>8.1f}",
                flush=True,
            )
            key = (result["config_name"], result["agent_name"])
            collected.setdefault(key, []).append(result["eval_result"])
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_episode_task, t): t for t in tasks}
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                print(
                    f"  [{completed:>3}/{total}] "
                    f"{result['config_name']:<14} {result['agent_name']:<16} "
                    f"seed={result['seed']:<5} "
                    f"reward={result['eval_result']['total_reward']:>8.1f}",
                    flush=True,
                )
                key = (result["config_name"], result["agent_name"])
                collected.setdefault(key, []).append(result["eval_result"])

    # Aggregate results
    results = {}
    for (config_name, agent_name), eval_results in collected.items():
        if config_name not in results:
            results[config_name] = {}

        rewards = [e["total_reward"] for e in eval_results]
        brier_storms = [e["mean_brier_storm"] for e in eval_results]
        brier_equips = [e["mean_brier_equip"] for e in eval_results]
        detection_lags = [e["mean_detection_lag"] for e in eval_results]
        tool_gaps = [e["total_tool_use_gap"] for e in eval_results]
        inference_gaps = [e["total_inference_gap"] for e in eval_results]
        planning_gaps = [e["total_planning_gap"] for e in eval_results]

        if verify:
            for eval_result in eval_results:
                for step in eval_result["step_results"]:
                    gap_sum = (step["tool_use_gap"] + step["inference_gap"]
                               + step["planning_gap"])
                    expected = step["oracle_reward"] - step["actual_reward"]
                    if abs(gap_sum - expected) > 1e-10:
                        print(f"DECOMPOSITION FAILED: {config_name}/{agent_name} "
                              f"day={step['day']}")
                        all_pass = False

        results[config_name][agent_name] = {
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
            "brier_storm": np.mean(brier_storms),
            "brier_equip": np.mean(brier_equips),
            "detection_lag": np.mean(detection_lags),
            "tool_gap": np.mean(tool_gaps),
            "inference_gap": np.mean(inference_gaps),
            "planning_gap": np.mean(planning_gaps),
        }

    return results, all_pass


def print_comparison_table(results):
    """Print the full comparison table."""
    header = (
        f"{'Config':<14} {'Agent':<16} {'Reward':>10} "
        f"{'Brier(S)':>9} {'Brier(E)':>9} {'Det.Lag':>8} "
        f"{'ToolGap':>9} {'InfGap':>9} {'PlanGap':>9}"
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print(header)
    print(separator)

    for config_name in ABLATION_CONFIGS:
        for agent_name, _ in BASELINES:
            m = results[config_name][agent_name]
            det_lag = f"{m['detection_lag']:.1f}" if m["detection_lag"] != float("inf") else "inf"
            print(
                f"{config_name:<14} {agent_name:<16} "
                f"{m['reward_mean']:>7.1f}+{m['reward_std']:>4.1f} "
                f"{m['brier_storm']:>9.4f} {m['brier_equip']:>9.4f} "
                f"{det_lag:>8} "
                f"{m['tool_gap']:>9.1f} {m['inference_gap']:>9.1f} "
                f"{m['planning_gap']:>9.1f}"
            )
        print(separator)


def verify_ordering(results):
    """Verify baseline ordering holds under every ablation config.

    Random < NaivePattern < CausalLearner <= CausalReasoner <= Oracle.
    CausalLearner can match CausalReasoner on easy configs.
    CausalReasoner can match Oracle when sensors are clean enough.
    """
    agent_names = [name for name, _ in BASELINES]
    # Pairs that allow equality: CausalLearner<=CausalReasoner, CausalReasoner<=Oracle
    allow_equal = {(agent_names[-3], agent_names[-2]),
                   (agent_names[-2], agent_names[-1])}
    all_ok = True

    for config_name in ABLATION_CONFIGS:
        rewards = [results[config_name][name]["reward_mean"] for name in agent_names]
        for i in range(len(rewards) - 1):
            pair = (agent_names[i], agent_names[i + 1])
            if pair in allow_equal:
                if rewards[i] > rewards[i + 1]:
                    print(
                        f"ORDERING VIOLATION in {config_name}: "
                        f"{agent_names[i]} ({rewards[i]:.1f}) > "
                        f"{agent_names[i+1]} ({rewards[i+1]:.1f})"
                    )
                    all_ok = False
            else:
                if rewards[i] >= rewards[i + 1]:
                    print(
                        f"ORDERING VIOLATION in {config_name}: "
                        f"{agent_names[i]} ({rewards[i]:.1f}) >= "
                        f"{agent_names[i+1]} ({rewards[i+1]:.1f})"
                    )
                    all_ok = False

    return all_ok


def verify_tool_use_gaps(results):
    """Verify that NaivePatternMatcher/Random have positive tool_use_gap
    and CausalReasoner/Oracle have ~0 tool_use_gap in 'full' config."""
    all_ok = True
    full = results.get("full", {})

    # Agents that don't use SQL should have positive tool_use_gap
    for name in ["Random", "NaivePattern"]:
        if name in full:
            gap = full[name]["tool_gap"]
            if gap <= 0:
                print(f"TOOL_GAP VIOLATION: {name} tool_gap={gap:.1f} should be > 0")
                all_ok = False

    # Agents that use SQL should have ~0 tool_use_gap
    for name in ["CausalLearner", "CausalReasoner", "Oracle"]:
        if name in full:
            gap = full[name]["tool_gap"]
            if abs(gap) > 1.0:
                print(f"TOOL_GAP VIOLATION: {name} tool_gap={gap:.1f} should be ~0")
                all_ok = False

    return all_ok


def main():
    """Run the full ablation suite and print results."""
    n_episodes = len(ABLATION_CONFIGS) * len(BASELINES) * len(DEFAULT_SEEDS)
    workers = os.cpu_count() or 1
    print(f"Running ablation suite: {len(ABLATION_CONFIGS)} configs x "
          f"{len(BASELINES)} baselines x {len(DEFAULT_SEEDS)} seeds = "
          f"{n_episodes} episodes ({workers} workers)")
    print("=" * 80)

    t0 = time.time()
    results, decomposition_ok = run_ablation_suite()
    elapsed = time.time() - t0

    print_comparison_table(results)

    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS:")
    print(f"  Decomposition identity: {'PASS' if decomposition_ok else 'FAIL'}")

    ordering_ok = verify_ordering(results)
    print(f"  Baseline ordering:      {'PASS' if ordering_ok else 'FAIL'}")

    tool_gap_ok = verify_tool_use_gaps(results)
    print(f"  Tool use gaps:          {'PASS' if tool_gap_ok else 'FAIL'}")

    all_ok = decomposition_ok and ordering_ok and tool_gap_ok
    if all_ok:
        print("\nAll verifications PASSED.")
    else:
        print("\nSome verifications FAILED. See above for details.")

    print(f"\nCompleted in {elapsed:.1f}s ({n_episodes/elapsed:.1f} episodes/s)")

    return results


if __name__ == "__main__":
    main()
