"""
Run LLM+Solver on 5 curriculum levels.

Setup:
  5 curriculum levels (0.0, 0.25, 0.50, 0.75, 1.0)
  1 agent (LLM+Solver, GPT-5.4 mini for speed)
  2 seeds per level (fast run) = 10 episodes total

Output: How LLM parameter discovery scales with curriculum difficulty.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import json
import time

from world_gen import curriculum_knobs, generate_config, validate_config

from fishing_game.simulator import FishingGameEnv
from fishing_game.evaluator import Evaluator
from fishing_game.llm_solver_agent import LLMSolverAgent
from fishing_game.traced_runner import run_llm_solver_episode

# Load .env from repo root parent directory
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-4o-mini"
CURRICULUM_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS_PER_LEVEL = [42, 123]


def run_episode(level, seed, cfg):
    """Run LLM+Solver on a curriculum level."""
    client = OpenAI(timeout=120.0)

    def gpt_fn(messages):
        resp = client.chat.completions.create(model=MODEL, messages=messages, timeout=120.0)
        return resp.choices[0].message.content

    agent = LLMSolverAgent(llm_fn=gpt_fn, config=cfg)

    result = run_llm_solver_episode(
        agent, seed=seed, config=cfg,
        save_path=f"traces/llm_solver_curriculum_L{level:.2f}_s{seed}.json"
    )

    return {
        "level": level,
        "seed": seed,
        "total_reward": result.get("total_reward", 0),
        "mean_brier_storm": result.get("mean_brier_storm", 0),
        "mean_brier_equip": result.get("mean_brier_equip", 0),
        "total_inference_gap": result.get("total_inference_gap", 0),
        "learned_config": agent._learned_config,
        "raw_response": agent._raw_model_response,
    }


def main():
    print("=" * 100)
    print(f"LLM+SOLVER ON CURRICULUM: {MODEL}")
    print("=" * 100)

    # Generate configs
    print("\n[Generating curriculum configs...]")
    configs = {}
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
            print(f"\n{'='*80}")
            print(f"Level {level:.2f}, Seed {seed}")
            print('='*80)

            try:
                t0 = time.time()
                result = run_episode(level, seed, cfg)
                elapsed = time.time() - t0

                results_by_level[level].append(result)

                print(f"\nResult: reward={result['total_reward']:.1f}, "
                      f"brier_s={result['mean_brier_storm']:.4f}, "
                      f"brier_e={result['mean_brier_equip']:.4f}, "
                      f"time={elapsed:.1f}s")

            except Exception as e:
                print(f"\nERROR: {e}")
                results_by_level[level].append({
                    "level": level,
                    "seed": seed,
                    "total_reward": 0,
                    "error": str(e)
                })

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"\n{'Level':<8} {'Reward1':<10} {'Reward2':<10} {'Mean':<10} {'Std':<10}")
    print("-" * 50)

    for level in CURRICULUM_LEVELS:
        results = results_by_level[level]
        rewards = [r.get("total_reward", 0) for r in results if "error" not in r]

        if len(rewards) == 2:
            mean_reward = sum(rewards) / len(rewards)
            std_reward = (abs(rewards[0] - rewards[1]) / 2) if len(rewards) > 1 else 0
            print(f"{level:<8.2f} {rewards[0]:<10.1f} {rewards[1]:<10.1f} {mean_reward:<10.1f} {std_reward:<10.1f}")
        else:
            print(f"{level:<8.2f} [INCOMPLETE - {len(rewards)} results]")

    # Save detailed results
    output_file = Path("reports/2026-04-curriculum/data/llm_solver_curriculum_results.json")
    with open(output_file, "w") as f:
        json.dump(results_by_level, f, indent=2, default=str)
    print(f"\nDetailed results saved to {output_file}")


if __name__ == "__main__":
    main()
