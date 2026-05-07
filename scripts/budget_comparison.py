"""Compare current budget formula vs Shanmugam-based budget across v1 ladder levels."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from causal_discovery import build_benchmark_instance  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_ladder import config_from_level, ladder_levels  # noqa: E402

SEEDS_PER_LEVEL = 6


def main() -> None:
    levels = ladder_levels()
    print(
        f"{'Lvl':>3} {'d':>2} {'k':>2} {'seed':>12} "
        f"{'k_undir':>7} {'opt_set':>7} {'slack':>5} "
        f"{'cur_budget':>10} {'new_budget(+1)':>14} {'new_budget(+2)':>14}"
    )
    print("-" * 100)

    rng = np.random.default_rng(42)
    for lvl_id, level in sorted(levels.items()):
        cfg = config_from_level(level)
        for trial in range(SEEDS_PER_LEVEL):
            seed = int(rng.integers(0, 2**31))
            try:
                inst = build_benchmark_instance(cfg, np.random.default_rng(seed))
            except Exception:
                continue

            k_undir = len(inst.observational_ceiling.undirected_edges)
            opt_size = len(inst.optimal_intervention_set)
            slack = level.budget_slack
            cur_budget = inst.intervention_budget  # = opt_size + slack
            new_budget_1 = opt_size + 1
            new_budget_2 = opt_size + 2

            print(
                f"{lvl_id:>3} {level.d:>2} {level.k:>2} {seed:>12} "
                f"{k_undir:>7} {opt_size:>7} {slack:>5} "
                f"{cur_budget:>10} {new_budget_1:>14} {new_budget_2:>14}"
            )


if __name__ == "__main__":
    main()
