"""Configuration models for the causal discovery benchmark."""

from causal_discovery.config.v1 import (
    BenchmarkConfig,
    DEFAULT_WEIGHT_RANGE,
    WeightInterval,
    WeightRange,
    make_v1_config,
)

__all__ = [
    "BenchmarkConfig",
    "DEFAULT_WEIGHT_RANGE",
    "WeightInterval",
    "WeightRange",
    "make_v1_config",
]
