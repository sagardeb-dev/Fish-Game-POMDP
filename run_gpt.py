"""Run GPT-5.4 agent on the Fishing Game with full tracing."""

from fishing_game.gpt_agent import GPTAgent
from fishing_game.traced_runner import run_traced_episode, print_traced_episode


if __name__ == "__main__":
    print("Running GPT-5.4 on Fishing Game (seed=42)...")
    print()

    output = run_traced_episode(
        agent_cls=GPTAgent,
        seed=42,
        save_path="traces/gpt54_seed42.json",
    )

    print_traced_episode(output)
    print(f"\nTrace saved to traces/gpt54_seed42.json")
