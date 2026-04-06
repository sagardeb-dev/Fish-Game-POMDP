"""
Run LLMAgent (GPT 5.4) and LLM+Solver (GPT 5.4) on 5 seeds.
Prints results table and appends to benchmark_results.md.

Usage:
    uv run python run_llm_benchmark.py
"""

import json
import time
import numpy as np
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from fishing_game.config import CONFIG
from fishing_game.simulator import FishingGameEnv
from fishing_game.evaluator import Evaluator
from fishing_game.gpt_agent import GPTAgent
from fishing_game.llm_solver_agent import LLMSolverAgent
from fishing_game.traced_runner import run_traced_episode, run_llm_solver_episode

# Load .env
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

SEEDS = [42, 123, 456, 789, 1024]
MODEL = "gpt-5.4"


def run_llm_agent_episode(seed):
    """Run LLMAgent (tool-calling GPT) for one episode."""
    agent_cls = lambda config=None: GPTAgent(model=MODEL, config=config)
    output = run_traced_episode(
        agent_cls=agent_cls, seed=seed, config=CONFIG,
        save_path=f"traces/llm_agent_{MODEL.replace('.','')}_seed{seed}.json",
    )
    return output["evaluation"]["total_reward"]


def run_llm_solver_ep(seed):
    """Run LLM+Solver for one episode."""
    client = OpenAI(timeout=120.0)

    def gpt_fn(messages):
        resp = client.chat.completions.create(model=MODEL, messages=messages)
        return resp.choices[0].message.content

    agent = LLMSolverAgent(llm_fn=gpt_fn, config=CONFIG)
    result = run_llm_solver_episode(
        agent, seed=seed, config=CONFIG,
        save_path=f"traces/llm_solver_{MODEL.replace('.','')}_seed{seed}.json",
    )
    return result["total_reward"]


def main():
    print(f"Running LLM benchmark: {MODEL}, {len(SEEDS)} seeds")
    print(f"Config: BENCHMARK_CONFIG (sensor_zones_per_step={CONFIG.get('sensor_zones_per_step', 4)})")
    print("=" * 70)

    results = {}

    # LLM+Solver
    print("\n--- LLM+Solver ---")
    solver_rewards = []
    for seed in SEEDS:
        print(f"\n[Seed {seed}]")
        t0 = time.time()
        reward = run_llm_solver_ep(seed)
        elapsed = time.time() - t0
        solver_rewards.append(reward)
        print(f"  Reward: {reward:.0f} ({elapsed:.1f}s)")
    results["LLM+Solver"] = solver_rewards

    # LLMAgent
    print("\n--- LLMAgent ---")
    agent_rewards = []
    for seed in SEEDS:
        print(f"\n[Seed {seed}]")
        t0 = time.time()
        reward = run_llm_agent_episode(seed)
        elapsed = time.time() - t0
        agent_rewards.append(reward)
        print(f"  Reward: {reward:.0f} ({elapsed:.1f}s)")
    results["LLMAgent"] = agent_rewards

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Agent':<16} {'Mean':>8} {'Std':>6}  Per-seed")
    print("-" * 70)
    for name in ["LLMAgent", "LLM+Solver"]:
        rewards = results[name]
        per_seed = "  ".join(f"s{s}={r:.0f}" for s, r in zip(SEEDS, rewards))
        print(f"{name:<16} {np.mean(rewards):>8.1f} {np.std(rewards):>5.1f}  {per_seed}")

    # Append to benchmark_results.md
    md_path = Path("benchmark_results.md")
    if md_path.exists():
        content = md_path.read_text()
        # Replace placeholder lines
        for name in ["LLMAgent", "LLM+Solver"]:
            rewards = results[name]
            per_seed = "  ".join(f"s{s}={r:.0f}" for s, r in zip(SEEDS, rewards))
            new_line = f"{name:<16} {np.mean(rewards):>8.1f} {np.std(rewards):>5.1f}  {per_seed}"
            content = content.replace(
                f"{name:<16} {'???':>8} {'???':>6}  (run with run_llm_benchmark.py)",
                new_line,
            )
        md_path.write_text(content)
        print(f"\nUpdated {md_path}")


if __name__ == "__main__":
    main()
