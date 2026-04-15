"""
Run the curriculum with the canonical mock LLM+Solver path.

This script intentionally uses the same LLMSolverAgent execution path as the
real benchmark. The only mocked part is the model response itself.
"""

import numpy as np

from world_gen import curriculum_knobs, generate_config, validate_config

from fishing_game.llm_solver_agent import MockLLMSolverAgent
from fishing_game.traced_runner import run_llm_solver_episode


def run_episode_mock(level, seed, cfg):
    """Run the stable mock LLM+Solver path on one curriculum level."""
    agent = MockLLMSolverAgent(config=cfg)
    eval_result = run_llm_solver_episode(agent, seed=seed, config=cfg)
    return {
        "level": level,
        "seed": seed,
        "total_reward": eval_result.get("total_reward", 0),
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
