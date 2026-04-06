"""Run LLM+Solver agent with GPT 5.4 on the Fishing Game."""

import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from fishing_game.config import CONFIG
from fishing_game.llm_solver_agent import LLMSolverAgent
from fishing_game.traced_runner import run_llm_solver_episode

# Load .env
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

client = OpenAI(timeout=120.0)


def gpt_llm_fn(messages):
    """Call GPT 5.4 with messages, return text response."""
    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=messages,
    )
    return response.choices[0].message.content


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    save_path = f"traces/llm_solver_gpt54_seed{seed}.json"

    print(f"Running LLM+Solver (GPT 5.4) seed={seed}")
    print("=" * 60)

    agent = LLMSolverAgent(llm_fn=gpt_llm_fn, config=CONFIG)
    run_llm_solver_episode(agent, seed=seed, save_path=save_path)


if __name__ == "__main__":
    main()
