"""V1 benchmark configuration."""

from __future__ import annotations

from dataclasses import dataclass


WeightInterval = tuple[float, float]
WeightRange = tuple[WeightInterval, WeightInterval]

DEFAULT_WEIGHT_RANGE: WeightRange = ((-2.0, -0.5), (0.5, 2.0))


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Serializable configuration for a causal discovery benchmark instance."""

    d: int
    k: int | None = None
    n_obs: int = 1000
    n_int: int | None = None
    weight_range: WeightRange = DEFAULT_WEIGHT_RANGE
    noise_var: float = 1.0
    faithfulness_eps: float = 0.1
    budget_slack: int = 0

    def __post_init__(self) -> None:
        if self.d < 4:
            raise ValueError(f"d must be >= 4 for v1 benchmark, got {self.d}")
        if self.n_obs <= 0:
            raise ValueError(f"n_obs must be positive, got {self.n_obs}")
        if self.k is None:
            object.__setattr__(self, "k", self.d + 1)
        if self.n_int is None:
            object.__setattr__(self, "n_int", self.n_obs)
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        max_edges = self.d * (self.d - 1) // 2
        if self.k > max_edges:
            raise ValueError(f"k must be <= {max_edges} for d={self.d}, got {self.k}")
        if self.n_int <= 0:
            raise ValueError(f"n_int must be positive, got {self.n_int}")
        if self.noise_var <= 0.0:
            raise ValueError(f"noise_var must be positive, got {self.noise_var}")
        if self.faithfulness_eps < 0.0:
            raise ValueError(
                f"faithfulness_eps must be non-negative, got {self.faithfulness_eps}"
            )
        if self.budget_slack < 0:
            raise ValueError(f"budget_slack must be non-negative, got {self.budget_slack}")

        neg_range, pos_range = self.weight_range
        if not (neg_range[0] < neg_range[1] < 0.0):
            raise ValueError(
                "weight_range negative interval must satisfy low < high < 0, "
                f"got {neg_range}"
            )
        if not (0.0 < pos_range[0] < pos_range[1]):
            raise ValueError(
                "weight_range positive interval must satisfy 0 < low < high, "
                f"got {pos_range}"
            )


def make_v1_config(d: int, **overrides: object) -> BenchmarkConfig:
    """Create a V1 benchmark config with canonical defaults."""
    return BenchmarkConfig(d=d, **overrides)
