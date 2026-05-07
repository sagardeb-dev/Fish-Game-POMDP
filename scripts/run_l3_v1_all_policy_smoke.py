"""Run one L3 v1 smoke instance across every current policy."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = "openrouter/deepseek/deepseek-v3.2"
DEFAULT_OUT_DIR = "traces/ladder/l3_v1_all_policy_smoke_deepseek_v32"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_env = root.parent / ".env"
    parser = argparse.ArgumentParser(
        description="Run one L3 v1 benchmark instance across all current policies."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--env-file", default=str(default_env))
    parser.add_argument("--max-steps-raw", default="20")
    parser.add_argument("--max-steps-stats", default="40")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(root / "run_ladder.py"),
        "--levels",
        "3",
        "--seeds-per-level",
        "1",
        "--models",
        args.model,
        "--out-dir",
        args.out_dir,
        "--env-file",
        args.env_file,
        "--max-steps-raw",
        str(args.max_steps_raw),
        "--max-steps-stats",
        str(args.max_steps_stats),
    ]
    print("[run]", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=root))


if __name__ == "__main__":
    main()
