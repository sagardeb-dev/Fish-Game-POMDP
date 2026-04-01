"""
Ablation runner for the Fishing Game.

6 ablation configs × 5 baselines × 5 seeds = 150 episodes.
Prints comparison table. Verifies baseline ordering and decomposition identity.
"""

import random as stdlib_random
from fishing_game.config import CONFIG
from fishing_game.simulator import FishingGameEnv
from fishing_game.evaluator import Evaluator
from fishing_game.baselines import (
    RandomAgent, NoToolsHeuristic, SearchOnlyHeuristic,
    BeliefAwareBaseline, OracleAgent,
)


# --- Ablation configurations ---
ABLATION_CONFIGS = {
    "full": {
        "check_weather_reports": True, "query_fishing_log": True,
        "analyze_data": True, "evaluate_options": True,
        "forecast_scenario": True, "read_barometer": True, "read_buoy": True,
    },
    "no_search": {
        "check_weather_reports": False, "query_fishing_log": True,
        "analyze_data": True, "evaluate_options": True,
        "forecast_scenario": True, "read_barometer": True, "read_buoy": True,
    },
    "search_only": {
        "check_weather_reports": True, "query_fishing_log": False,
        "analyze_data": False, "evaluate_options": False,
        "forecast_scenario": False, "read_barometer": False, "read_buoy": False,
    },
    "no_optimizer": {
        "check_weather_reports": True, "query_fishing_log": True,
        "analyze_data": True, "evaluate_options": False,
        "forecast_scenario": True, "read_barometer": True, "read_buoy": True,
    },
    "no_whatif": {
        "check_weather_reports": True, "query_fishing_log": True,
        "analyze_data": True, "evaluate_options": True,
        "forecast_scenario": False, "read_barometer": True, "read_buoy": True,
    },
    "no_tools": {
        "check_weather_reports": False, "query_fishing_log": False,
        "analyze_data": False, "evaluate_options": False,
        "forecast_scenario": False, "read_barometer": False, "read_buoy": False,
    },
}

BASELINES = [
    ("Random", RandomAgent),
    ("NoTools", NoToolsHeuristic),
    ("SearchOnly", SearchOnlyHeuristic),
    ("BeliefAware", BeliefAwareBaseline),
    ("Oracle", OracleAgent),
]

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


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


def run_ablation_suite(seeds=None, config=None, verify=True):
    """
    Run all baselines × all ablation configs × all seeds.
    Returns nested dict: results[config_name][agent_name] = {metrics...}
    """
    seeds = seeds or DEFAULT_SEEDS
    cfg = config or CONFIG
    results = {}
    all_pass = True

    for config_name, ablation in ABLATION_CONFIGS.items():
        results[config_name] = {}
        for agent_name, agent_cls in BASELINES:
            rewards = []
            brier_storms = []
            brier_zones = []
            detection_lags = []
            tool_gaps = []
            inference_gaps = []
            planning_gaps = []

            for seed in seeds:
                total_reward, trace, eval_result = run_episode(
                    agent_cls, seed, config=cfg, ablation=ablation,
                )

                rewards.append(eval_result["total_reward"])
                brier_storms.append(eval_result["mean_brier_storm"])
                brier_zones.append(eval_result["mean_brier_zone"])
                detection_lags.append(eval_result["mean_detection_lag"])
                tool_gaps.append(eval_result["total_tool_use_gap"])
                inference_gaps.append(eval_result["total_inference_gap"])
                planning_gaps.append(eval_result["total_planning_gap"])

                # Verify decomposition identity
                if verify:
                    for step in eval_result["step_results"]:
                        gap_sum = (step["tool_use_gap"] + step["inference_gap"]
                                   + step["planning_gap"])
                        expected = step["oracle_reward"] - step["actual_reward"]
                        if abs(gap_sum - expected) > 1e-10:
                            print(f"DECOMPOSITION FAILED: {config_name}/{agent_name} "
                                  f"seed={seed} day={step['day']}")
                            all_pass = False

            import numpy as np
            results[config_name][agent_name] = {
                "reward_mean": np.mean(rewards),
                "reward_std": np.std(rewards),
                "brier_storm": np.mean(brier_storms),
                "brier_zone": np.mean(brier_zones),
                "detection_lag": np.mean(detection_lags),
                "tool_gap": np.mean(tool_gaps),
                "inference_gap": np.mean(inference_gaps),
                "planning_gap": np.mean(planning_gaps),
            }

    return results, all_pass


def print_comparison_table(results):
    """Print the full comparison table."""
    header = (
        f"{'Config':<14} {'Agent':<14} {'Reward':>10} "
        f"{'Brier(S)':>9} {'Brier(Z)':>9} {'Det.Lag':>8} "
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
                f"{config_name:<14} {agent_name:<14} "
                f"{m['reward_mean']:>7.1f}±{m['reward_std']:>4.1f} "
                f"{m['brier_storm']:>9.4f} {m['brier_zone']:>9.4f} "
                f"{det_lag:>8} "
                f"{m['tool_gap']:>9.1f} {m['inference_gap']:>9.1f} "
                f"{m['planning_gap']:>9.1f}"
            )
        print(separator)


def verify_ordering(results):
    """Verify baseline ordering holds under every ablation config."""
    agent_names = [name for name, _ in BASELINES]
    all_ok = True

    for config_name in ABLATION_CONFIGS:
        rewards = [results[config_name][name]["reward_mean"] for name in agent_names]
        for i in range(len(rewards) - 1):
            if rewards[i] >= rewards[i + 1]:
                print(
                    f"ORDERING VIOLATION in {config_name}: "
                    f"{agent_names[i]} ({rewards[i]:.1f}) >= "
                    f"{agent_names[i+1]} ({rewards[i+1]:.1f})"
                )
                all_ok = False

    return all_ok


def main():
    """Run the full ablation suite and print results."""
    print("Running ablation suite: 6 configs × 5 baselines × 5 seeds = 150 episodes")
    print("=" * 80)

    results, decomposition_ok = run_ablation_suite()
    print_comparison_table(results)

    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS:")
    print(f"  Decomposition identity: {'PASS' if decomposition_ok else 'FAIL'}")

    ordering_ok = verify_ordering(results)
    print(f"  Baseline ordering:      {'PASS' if ordering_ok else 'FAIL'}")

    if decomposition_ok and ordering_ok:
        print("\nAll verifications PASSED.")
    else:
        print("\nSome verifications FAILED. See above for details.")

    return results


if __name__ == "__main__":
    main()
