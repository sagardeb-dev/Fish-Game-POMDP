"""Core benchmark-owned data structures."""

from causal_discovery.core.dag import DAG
from causal_discovery.core.permutation import Permutation
from causal_discovery.core.scm import LinearGaussianSCM

__all__ = [
    "DAG",
    "LinearGaussianSCM",
    "Permutation",
]
