"""Core package for the causal discovery benchmark."""

from causal_discovery.benchmark import BenchmarkInstance, build_benchmark_instance
from causal_discovery.config import (
    BenchmarkConfig,
    DEFAULT_WEIGHT_RANGE,
    WeightInterval,
    WeightRange,
    make_v1_config,
)
from causal_discovery.core import DAG, LinearGaussianSCM, Permutation
from causal_discovery.equivalence import (
    CPDAG,
    compute_minimum_intervention_set,
    dag_to_cpdag,
    reject_graph,
    reject_intervention_profile,
    reject_scm,
)
from causal_discovery.graph_gen import (
    sample_random_dag,
    sample_random_topological_order,
    valid_forward_edges,
)
from causal_discovery.scm import (
    implied_covariance,
    is_near_singular,
    parameterize_linear_gaussian_scm,
    partial_correlation,
    relevant_structural_partial_correlations,
    sample_edge_weights,
    sample_noise_variances,
    scm_weight_matrix,
    topological_order_from_dag,
)
from causal_discovery.sampling import sample_interventional_data, sample_observational_data

__all__ = [
    "BenchmarkConfig",
    "BenchmarkInstance",
    "CPDAG",
    "DAG",
    "DEFAULT_WEIGHT_RANGE",
    "LinearGaussianSCM",
    "Permutation",
    "WeightInterval",
    "WeightRange",
    "build_benchmark_instance",
    "compute_minimum_intervention_set",
    "dag_to_cpdag",
    "implied_covariance",
    "is_near_singular",
    "make_v1_config",
    "parameterize_linear_gaussian_scm",
    "partial_correlation",
    "relevant_structural_partial_correlations",
    "reject_graph",
    "reject_intervention_profile",
    "reject_scm",
    "sample_edge_weights",
    "sample_interventional_data",
    "sample_noise_variances",
    "sample_observational_data",
    "sample_random_dag",
    "sample_random_topological_order",
    "scm_weight_matrix",
    "topological_order_from_dag",
    "valid_forward_edges",
]
