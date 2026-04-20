"""Benchmark instance assembly for active causal discovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal_discovery.config import BenchmarkConfig
from causal_discovery.core import DAG, LinearGaussianSCM, Permutation
from causal_discovery.equivalence import (
    CPDAG,
    compute_minimum_intervention_set,
    dag_to_cpdag,
    reject_graph,
    reject_intervention_profile,
    reject_scm,
)
from causal_discovery.graph_gen import sample_random_dag
from causal_discovery.sampling import sample_observational_data
from causal_discovery.scm import parameterize_linear_gaussian_scm


@dataclass(frozen=True, slots=True)
class BenchmarkInstance:
    """One accepted benchmark world with public-facing labels."""

    config: BenchmarkConfig
    true_dag: DAG
    observational_ceiling: CPDAG
    scm: LinearGaussianSCM
    optimal_intervention_set: tuple[int, ...]
    observational_data: np.ndarray
    intervention_budget: int
    label_permutation: Permutation


def build_benchmark_instance(
    config: BenchmarkConfig,
    rng: np.random.Generator,
    max_attempts: int = 10_000,
) -> BenchmarkInstance:
    """Build one accepted benchmark instance under the v1 rejection policy."""
    if max_attempts <= 0:
        raise ValueError(f"max_attempts must be positive, got {max_attempts}")

    for _ in range(max_attempts):
        dag_internal = sample_random_dag(config.d, config.k, rng)
        cpdag_internal = dag_to_cpdag(dag_internal)
        if reject_graph(dag_internal, cpdag_internal):
            continue

        scm_internal = parameterize_linear_gaussian_scm(
            dag_internal,
            config.weight_range,
            config.noise_var,
            rng,
        )
        if reject_scm(scm_internal, config.faithfulness_eps):
            continue

        intervention_set_internal = compute_minimum_intervention_set(
            dag_internal,
            cpdag_internal,
        )
        if reject_intervention_profile(intervention_set_internal, cpdag_internal):
            continue

        permutation = Permutation.from_mapping(rng.permutation(config.d))
        dag_public = dag_internal.relabel(permutation)
        cpdag_public = cpdag_internal.relabel(permutation)
        scm_public = scm_internal.relabel(permutation)
        intervention_set_public = tuple(
            sorted(permutation.apply_node(node) for node in intervention_set_internal)
        )
        observational_data = sample_observational_data(scm_public, config.n_obs, rng)
        budget = len(intervention_set_public) + config.budget_slack

        return BenchmarkInstance(
            config=config,
            true_dag=dag_public,
            observational_ceiling=cpdag_public,
            scm=scm_public,
            optimal_intervention_set=intervention_set_public,
            observational_data=observational_data,
            intervention_budget=budget,
            label_permutation=permutation,
        )

    raise RuntimeError(
        "Unable to build an accepted benchmark instance after "
        f"{max_attempts} attempts"
    )

