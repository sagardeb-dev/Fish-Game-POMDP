"""
Run LLMAgent, LLM+Solver, and CodingAgent (GPT 5.4) on 5 seeds.
Prints results table with full metrics and updates docs/reports/benchmark_results.md.

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
from fishing_game.coding_agent import CodingAgent
from fishing_game.traced_runner import run_traced_episode, run_llm_solver_episode

ROOT = Path(__file__).resolve().parent
TRACES_DIR = ROOT / "traces"
BENCHMARK_RESULTS_PATH = ROOT / "docs" / "reports" / "benchmark_results.md"

# Load .env
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

SEEDS = [42, 123, 456, 789, 1024]
MODEL = "gpt-5.4"


def run_llm_agent_episode(seed):
    """Run LLMAgent (tool-calling GPT) for one episode. Returns full evaluation dict."""
    agent_cls = lambda config=None: GPTAgent(model=MODEL, config=config)
    output = run_traced_episode(
        agent_cls=agent_cls, seed=seed, config=CONFIG,
        save_path=str(TRACES_DIR / f"llm_agent_{MODEL.replace('.','')}_seed{seed}.json"),
    )
    return output["evaluation"]


def run_llm_solver_ep(seed):
    """Run LLM+Solver for one episode. Returns full evaluation dict."""
    client = OpenAI(timeout=120.0)

    def gpt_fn(messages):
        resp = client.chat.completions.create(model=MODEL, messages=messages)
        return resp.choices[0].message.content

    agent = LLMSolverAgent(llm_fn=gpt_fn, config=CONFIG)
    result = run_llm_solver_episode(
        agent, seed=seed, config=CONFIG,
        save_path=str(TRACES_DIR / f"llm_solver_{MODEL.replace('.','')}_seed{seed}.json"),
    )
    return result


def run_coding_agent_ep(seed):
    """Run CodingAgent for one episode. Returns full evaluation dict."""
    env = FishingGameEnv(config=CONFIG)
    obs = env.reset(seed=seed)

    agent = CodingAgent(model=MODEL, config=CONFIG)
    agent.reset()

    for day in range(CONFIG["episode_length"]):
        result = agent.act(env, obs)
        if result["done"]:
            break
        obs = result["observation"]

    # Save trace
    trace = env.get_trace()
    model_slug = MODEL.replace(".", "")
    save_path = TRACES_DIR / f"coding_agent_{model_slug}_seed{seed}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(trace, f, indent=2, default=str)

    # Evaluate
    evaluator = Evaluator(config=CONFIG)
    return evaluator.evaluate_episode(trace)


def main():
    print(f"Running LLM benchmark: {MODEL}, {len(SEEDS)} seeds")
    print(f"Config: BENCHMARK_CONFIG (sensor_zones_per_step={CONFIG.get('sensor_zones_per_step', 4)})")
    print("=" * 70)

    results = {}

    # LLM+Solver
    print("\n--- LLM+Solver ---")
    solver_evals = []
    for seed in SEEDS:
        print(f"\n[Seed {seed}]")
        t0 = time.time()
        ev = run_llm_solver_ep(seed)
        elapsed = time.time() - t0
        solver_evals.append(ev)
        print(f"  Reward: {ev['total_reward']:.0f}  Brier(S)={ev['mean_brier_storm']:.4f}  "
              f"Brier(E)={ev['mean_brier_equip']:.4f}  ({elapsed:.1f}s)")
    results["LLM+Solver"] = solver_evals

    # LLMAgent
    print("\n--- LLMAgent ---")
    agent_evals = []
    for seed in SEEDS:
        print(f"\n[Seed {seed}]")
        t0 = time.time()
        ev = run_llm_agent_episode(seed)
        elapsed = time.time() - t0
        agent_evals.append(ev)
        print(f"  Reward: {ev['total_reward']:.0f}  Brier(S)={ev['mean_brier_storm']:.4f}  "
              f"Brier(E)={ev['mean_brier_equip']:.4f}  ({elapsed:.1f}s)")
    results["LLMAgent"] = agent_evals

    # CodingAgent
    print("\n--- CodingAgent ---")
    coding_evals = []
    for seed in SEEDS:
        print(f"\n[Seed {seed}]")
        t0 = time.time()
        ev = run_coding_agent_ep(seed)
        elapsed = time.time() - t0
        coding_evals.append(ev)
        print(f"  Reward: {ev['total_reward']:.0f}  Brier(S)={ev['mean_brier_storm']:.4f}  "
              f"Brier(E)={ev['mean_brier_equip']:.4f}  ({elapsed:.1f}s)")
    results["CodingAgent"] = coding_evals

    # Print full summary
    print("\n" + "=" * 90)
    print("RESULTS")
    print("=" * 90)
    header = (f"{'Agent':<16} {'Reward':>10} {'Brier(S)':>9} {'Brier(E)':>9} "
              f"{'ToolGap':>9} {'InfGap':>9} {'PlanGap':>9}")
    print(header)
    print("-" * 90)

    agent_names = ["LLMAgent", "LLM+Solver", "CodingAgent"]
    for name in agent_names:
        evals = results[name]
        rewards = [e["total_reward"] for e in evals]
        brier_s = [e["mean_brier_storm"] for e in evals]
        brier_e = [e["mean_brier_equip"] for e in evals]
        tool_gaps = [e["total_tool_use_gap"] for e in evals]
        inf_gaps = [e["total_inference_gap"] for e in evals]
        plan_gaps = [e["total_planning_gap"] for e in evals]
        print(
            f"{name:<16} {np.mean(rewards):>7.1f}±{np.std(rewards):>4.0f} "
            f"{np.mean(brier_s):>9.4f} {np.mean(brier_e):>9.4f} "
            f"{np.mean(tool_gaps):>9.1f} {np.mean(inf_gaps):>9.1f} "
            f"{np.mean(plan_gaps):>9.1f}"
        )

    # Per-seed reward table
    print(f"\n{'Agent':<16} {'Mean':>8} {'Std':>6}  Per-seed")
    print("-" * 70)
    for name in agent_names:
        rewards = [e["total_reward"] for e in results[name]]
        per_seed = "  ".join(f"s{s}={r:.0f}" for s, r in zip(SEEDS, rewards))
        print(f"{name:<16} {np.mean(rewards):>8.1f} {np.std(rewards):>5.1f}  {per_seed}")

    # Update docs/reports/benchmark_results.md
    _update_benchmark_md(results)


def _update_benchmark_md(results):
    """Update the LLM section of docs/reports/benchmark_results.md."""
    md_path = BENCHMARK_RESULTS_PATH
    if not md_path.exists():
        return

    agent_names = ["LLMAgent", "LLM+Solver", "CodingAgent"]
    lines = []
    for name in agent_names:
        evals = results[name]
        rewards = [e["total_reward"] for e in evals]
        brier_s = [e["mean_brier_storm"] for e in evals]
        brier_e = [e["mean_brier_equip"] for e in evals]
        tool_gaps = [e["total_tool_use_gap"] for e in evals]
        inf_gaps = [e["total_inference_gap"] for e in evals]
        plan_gaps = [e["total_planning_gap"] for e in evals]
        per_seed = "  ".join(f"s{s}={r:.0f}" for s, r in zip(SEEDS, rewards))
        lines.append(
            f"{name:<16} {np.mean(rewards):>8.1f} {np.std(rewards):>5.1f}  "
            f"Brier(S)={np.mean(brier_s):.4f}  Brier(E)={np.mean(brier_e):.4f}  "
            f"ToolGap={np.mean(tool_gaps):.1f}  InfGap={np.mean(inf_gaps):.1f}  "
            f"PlanGap={np.mean(plan_gaps):.1f}\n"
            f"{'':16} {'':>8} {'':>6}  {per_seed}"
        )

    # Rebuild LLM section
    content = md_path.read_text()
    marker = "## LLM Agents"
    idx = content.find(marker)
    if idx == -1:
        content += f"\n\n{marker} (GPT 5.4)\n\n" + "\n".join(lines) + "\n"
    else:
        content = content[:idx] + (
            f"{marker} (GPT 5.4)\n\n"
            f"Agent                Mean    Std  Metrics\n"
            f"{'-'*80}\n"
            + "\n".join(lines) + "\n"
        )
    md_path.write_text(content)
    print(f"\nUpdated {md_path}")


if __name__ == "__main__":
    main()
