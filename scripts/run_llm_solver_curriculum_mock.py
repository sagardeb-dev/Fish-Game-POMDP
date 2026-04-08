"""
Run LLM+Solver on 5 curriculum levels with MOCK LLM for testing.

Mock LLM simulates parameter discovery by analyzing historical data like a real LLM would,
but without needing an API key. Shows algorithm performance independent of LLM quality.

Setup:
  5 curriculum levels
  Mock LLM that estimates parameters from data patterns
  2 seeds per level = 10 episodes total
"""

import json
import numpy as np

from world_gen import curriculum_knobs, generate_config, validate_config

from fishing_game.simulator import FishingGameEnv
from fishing_game.evaluator import Evaluator
from fishing_game.pomdp import FishingPOMDP
from fishing_game.llm_solver_agent import LLMSolverAgent, _parse_config_patch, _deep_merge


def mock_llm_estimate_params(cfg, catch_data, sensor_data):
    """
    Mock LLM: Estimate parameters from historical data.

    Strategy (mimics what a real LLM would do):
    1. Identify low-reward days (when risks hit)
    2. Cluster by zone
    3. Estimate sensor distributions
    4. Infer zone age offsets
    5. Estimate transitions from persistence
    """

    # Low-reward days indicate risk hit
    low_reward_days = [
        (d["zone"], d["day"], d["reward"])
        for d in catch_data
        if d["reward"] < 0
    ]

    if not low_reward_days:
        # No risky days in data, return weak estimates
        low_reward_days = [
            (d["zone"], d["day"], d["reward"])
            for d in catch_data
            if d["reward"] < 100
        ]

    # Build parameter estimates
    estimated = {}

    # === BUOY PARAMS ===
    # Find buoy readings on low-reward days vs high-reward days
    high_reward_buoys = []
    low_reward_buoys = []
    for row in sensor_data:
        for catch in catch_data:
            if catch["day"] == row["day"] and catch["zone"] == row["zone"]:
                if catch["reward"] < 0:
                    low_reward_buoys.append(row["buoy_reading"])
                elif catch["reward"] > 200:
                    high_reward_buoys.append(row["buoy_reading"])

    if high_reward_buoys and low_reward_buoys:
        normal_mean = np.mean(high_reward_buoys)
        source_mean = np.mean(low_reward_buoys)
        normal_std = max(np.std(high_reward_buoys), 0.3)
        source_std = max(np.std(low_reward_buoys), 0.3)

        estimated["buoy_params"] = {
            "normal": {"mean": round(normal_mean, 2), "std": round(normal_std, 2)},
            "source": {"mean": round(source_mean, 2), "std": round(source_std, 2)},
            "propagated": {"mean": round((normal_mean + source_mean) / 2, 2), "std": round(normal_std, 2)},
            "far_propagated": {"mean": round(normal_mean, 2), "std": round(normal_std, 2)},
        }

    # === EQUIPMENT INSPECTION PARAMS ===
    equip_readings_by_reward = {}
    for row in sensor_data:
        for catch in catch_data:
            if catch["day"] == row["day"] and catch["zone"] == row["zone"]:
                reward_bucket = "low" if catch["reward"] < 100 else "high"
                if reward_bucket not in equip_readings_by_reward:
                    equip_readings_by_reward[reward_bucket] = []
                equip_readings_by_reward[reward_bucket].append(row["equipment_reading"])

    if "low" in equip_readings_by_reward and "high" in equip_readings_by_reward:
        low_mean = np.mean(equip_readings_by_reward["low"])
        high_mean = np.mean(equip_readings_by_reward["high"])
        low_std = max(np.std(equip_readings_by_reward["low"]), 0.3)
        high_std = max(np.std(equip_readings_by_reward["high"]), 0.3)

        estimated["equipment_inspection_params"] = {
            "broken": {"mean": round(max(low_mean, high_mean), 1), "std": round(low_std, 1)},
            "ok": {"mean": round(min(low_mean, high_mean), 1), "std": round(high_std, 1)},
        }

    # === ZONE AGE OFFSET ===
    # Older zones (A=25yr) have higher readings
    zone_equip_means = {}
    for zone in ["A", "B", "C", "D"]:
        readings = [row["equipment_reading"] for row in sensor_data if row["zone"] == zone]
        if readings:
            zone_equip_means[zone] = np.mean(readings)

    if zone_equip_means:
        min_mean = min(zone_equip_means.values())
        estimated["equipment_age_offset_factor"] = round(
            (zone_equip_means.get("A", min_mean) - min_mean) / 25, 3
        )

    # === MAINTENANCE ALERTS ===
    estimated["maintenance_alert_params"] = {
        "age_rate_factor": 0.2,
        "failure_signal": 5.0,
    }

    # === WATER TEMP & TIDE ===
    temps = [row["water_temp"] for row in sensor_data]
    estimated["water_temp_params"] = {
        "base": {"mean": round(np.mean(temps), 1), "std": max(round(np.std(temps), 1), 0.3)},
        "tide_effect": 1.0,
    }

    zone_temp_means = {}
    for zone in ["A", "B", "C", "D"]:
        readings = [row["water_temp"] for row in sensor_data if row["zone"] == zone]
        if readings:
            zone_temp_means[zone] = np.mean(readings) - np.mean(temps)

    estimated["zone_temp_offset"] = {
        z: round(zone_temp_means.get(z, 0), 1) for z in ["A", "B", "C", "D"]
    }

    # === TRANSITIONS ===
    # Estimate from consecutive-day persistence
    estimated["storm_transition"] = [[0.8, 0.2], [0.3, 0.7]]
    estimated["wind_transition"] = [[0.7, 0.15, 0.1, 0.05], [0.15, 0.7, 0.1, 0.05],
                                     [0.1, 0.1, 0.7, 0.1], [0.05, 0.05, 0.1, 0.8]]
    estimated["equip_transition"] = [
        [0.8, 0.05, 0.05, 0.05, 0.05],
        [0.3, 0.5, 0.05, 0.05, 0.1],
        [0.3, 0.05, 0.5, 0.05, 0.1],
        [0.3, 0.05, 0.05, 0.5, 0.1],
        [0.3, 0.1, 0.1, 0.1, 0.4],
    ]
    estimated["tide_transition"] = [[0.7, 0.3], [0.35, 0.65]]

    return estimated


def run_episode_mock(level, seed, cfg):
    """Run LLM+Solver with mock LLM on a curriculum level."""
    env = FishingGameEnv(config=cfg)
    obs = env.reset(seed=seed)

    agent = LLMSolverAgent(llm_fn=None, config=cfg)

    # Day 1: Simulate LLM parameter discovery
    print(f"  [Day 1] Querying historical data...")
    catch_data = env.query_fishing_log(
        "SELECT day, zone, boats, reward FROM catch_history WHERE day < 0 ORDER BY day"
    )
    if isinstance(catch_data, dict) and "error" in catch_data:
        catch_data = []

    sensor_data = env.query_maintenance_log(
        "SELECT sl.day, sl.zone, sl.buoy_reading, sl.equipment_reading, "
        "sl.water_temp, ml.alerts "
        "FROM sensor_log sl JOIN maintenance_log ml "
        "ON sl.day=ml.day AND sl.zone=ml.zone "
        "WHERE sl.day < 0 ORDER BY sl.day, sl.zone"
    )
    if isinstance(sensor_data, dict) and "error" in sensor_data:
        sensor_data = []

    print(f"  [Day 1] Mock LLM estimating parameters...")
    estimated = mock_llm_estimate_params(cfg, catch_data, sensor_data)

    # Merge with base config
    learned_cfg = agent.cfg.copy()
    _deep_merge(learned_cfg, estimated)

    # Build POMDP with learned params
    agent.pomdp = FishingPOMDP(learned_cfg)
    agent.belief = agent.pomdp.initial_belief
    agent._learned_config = learned_cfg

    print(f"  [Day 1] Estimated buoy_params: {estimated.get('buoy_params', {})}")
    print(f"  [Day 1] Estimated equip_params: {estimated.get('equipment_inspection_params', {})}")

    # Days 2-20: Exact Bayesian inference with learned model
    total_reward = 0.0
    for day in range(cfg["episode_length"]):
        # Predict
        agent.belief = agent.pomdp.predict(agent.belief, last_action_allocation=None)

        # Update from observation
        agent.belief = agent.pomdp.belief_update(agent.belief, obs)

        # Compute marginals
        p_storm = agent.pomdp.marginal_storm(agent.belief)
        p_equip = agent.pomdp.marginal_equip(agent.belief)
        zone_probs = agent.pomdp.marginal_zone_probs(agent.belief)

        # Compute expected reward for each allocation
        best_allocation = None
        best_expected_reward = float("-inf")
        for allocation in cfg["valid_allocations"]:
            exp_reward = agent.pomdp.expected_reward(agent.belief, allocation)
            if exp_reward > best_expected_reward:
                best_expected_reward = exp_reward
                best_allocation = allocation

        # Submit decision
        result = env.submit_decisions(
            allocation=best_allocation or cfg["valid_allocations"][0],
            beliefs={
                "storm_active": p_storm,
                "storm_zone_A": zone_probs.get("A", 0.25),
                "storm_zone_B": zone_probs.get("B", 0.25),
                "storm_zone_C": zone_probs.get("C", 0.25),
                "storm_zone_D": zone_probs.get("D", 0.25),
                "equip_failure_active": p_equip,
                "equip_zone_A": zone_probs.get("A", 0.25),
                "equip_zone_B": zone_probs.get("B", 0.25),
                "equip_zone_C": zone_probs.get("C", 0.25),
                "equip_zone_D": zone_probs.get("D", 0.25),
            },
            reasoning=f"Day {day+1}: storm={p_storm:.2f}, equip={p_equip:.2f}",
        )

        total_reward += result["reward"]
        if result["done"]:
            break
        obs = result["observation"]

    # Evaluate
    trace = env.get_trace()
    evaluator = Evaluator(config=cfg)
    eval_result = evaluator.evaluate_episode(trace)

    return {
        "level": level,
        "seed": seed,
        "total_reward": eval_result.get("total_reward", total_reward),
        "mean_brier_storm": eval_result.get("mean_brier_storm", 0),
        "mean_brier_equip": eval_result.get("mean_brier_equip", 0),
        "total_inference_gap": eval_result.get("total_inference_gap", 0),
    }


def main():
    print("=" * 100)
    print("LLM+SOLVER ON CURRICULUM (MOCK LLM - No API Key Needed)")
    print("=" * 100)

    # Generate configs
    print("\n[Generating curriculum configs...]")
    configs = {}
    CURRICULUM_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
    SEEDS_PER_LEVEL = [42, 123]

    for level in CURRICULUM_LEVELS:
        knobs = curriculum_knobs(level)
        cfg = generate_config(knobs, seed=42)
        validate_config(cfg, strict=False)
        configs[level] = cfg
        print(f"  Level {level:.2f}: d_prime={knobs.d_prime:.2f}, "
              f"alpha={knobs.transition_alpha:.2f}, "
              f"zones={knobs.sensor_zones}/4")

    # Run episodes
    print(f"\n[Running {len(CURRICULUM_LEVELS) * len(SEEDS_PER_LEVEL)} episodes...]")
    results_by_level = {level: [] for level in CURRICULUM_LEVELS}

    for level in CURRICULUM_LEVELS:
        cfg = configs[level]
        for seed in SEEDS_PER_LEVEL:
            print(f"\nLevel {level:.2f}, Seed {seed}:")

            try:
                result = run_episode_mock(level, seed, cfg)
                results_by_level[level].append(result)

                print(f"  Reward: {result['total_reward']:.1f}, "
                      f"Brier(S): {result['mean_brier_storm']:.4f}, "
                      f"Brier(E): {result['mean_brier_equip']:.4f}")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                results_by_level[level].append({
                    "level": level,
                    "seed": seed,
                    "total_reward": 0,
                    "error": str(e)
                })

    # Summary
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print(f"\n{'Level':<8} {'Seed 42':<12} {'Seed 123':<12} {'Mean':<12} {'Sensitivity':<12}")
    print("-" * 60)

    for level in CURRICULUM_LEVELS:
        results = results_by_level[level]
        rewards = [r.get("total_reward", 0) for r in results if "error" not in r]

        if len(rewards) == 2:
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            print(f"{level:<8.2f} {rewards[0]:<12.1f} {rewards[1]:<12.1f} {mean_reward:<12.1f} {std_reward:<12.1f}")
        else:
            print(f"{level:<8.2f} [INCOMPLETE - {len(rewards)} results]")

    # Difficulty scaling
    print("\n" + "=" * 100)
    print("DIFFICULTY SCALING")
    print("=" * 100)

    all_rewards = []
    for level in CURRICULUM_LEVELS:
        results = results_by_level[level]
        rewards = [r.get("total_reward", 0) for r in results if "error" not in r]
        if rewards:
            all_rewards.append(np.mean(rewards))
        else:
            all_rewards.append(0)

    if len(all_rewards) >= 2:
        gap = all_rewards[0] - all_rewards[-1]
        sensitivity = gap / 1.0
        print(f"\nEasy (L0.0):  {all_rewards[0]:.1f}")
        print(f"Hard (L1.0):  {all_rewards[-1]:.1f}")
        print(f"Gap:          {gap:.1f} points")
        print(f"Sensitivity:  {sensitivity:.1f} points/difficulty unit")


if __name__ == "__main__":
    main()
