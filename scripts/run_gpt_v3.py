"""Run GPT agent on Fishing Game v4 with full tracing."""

import sys
from pathlib import Path
from fishing_game.config import CONFIG, HARD_CONFIG
from fishing_game.gpt_agent import GPTAgent
from fishing_game.traced_runner import run_traced_episode, print_traced_episode

TRACES_DIR = Path(__file__).resolve().parents[1] / "traces"


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "easy"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    model = sys.argv[3] if len(sys.argv) > 3 else "gpt-5.4"

    config = HARD_CONFIG if mode == "hard" else CONFIG
    agent_cls = lambda config=None: GPTAgent(model=model, config=config)

    model_slug = model.replace("/", "_").replace(" ", "_")
    difficulty = "hard" if mode == "hard" else "easy"
    save_path = str(TRACES_DIR / f"{model_slug}_v4_{difficulty}_seed{seed}.json")

    print(f"Running {model} on Fishing Game v4 ({mode} mode, seed={seed})...")
    print()

    output = run_traced_episode(
        agent_cls=agent_cls,
        seed=seed,
        config=config,
        save_path=save_path,
    )

    print_traced_episode(output)
    print(f"\nTrace saved to {save_path}")
