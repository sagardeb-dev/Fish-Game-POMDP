import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from run_random_dag_baseline import density_probe_levels, make_random_uniform_submission


def test_random_uniform_submission_is_blind_directed_dag():
    submission = make_random_uniform_submission(5, np.random.default_rng(123))

    assert submission.num_nodes == 5
    assert len(submission.undirected_edges) == 0
    assert submission.interventions_used == 0
    assert 0 <= len(submission.directed_edges) <= 10

    for src, dst in submission.directed_edges:
        assert src != dst
        assert 0 <= src < 5
        assert 0 <= dst < 5


def test_random_uniform_submission_does_not_force_true_edge_count():
    edge_counts = {
        len(make_random_uniform_submission(5, np.random.default_rng(seed)).directed_edges)
        for seed in range(30)
    }

    assert all(0 <= count <= 10 for count in edge_counts)
    assert any(count != 6 for count in edge_counts)
    assert len(edge_counts) > 1


def test_density_probe_catalog_matches_candidate_ladder():
    levels = density_probe_levels()

    assert list(levels) == [0, 1, 2, 3, 4, 5]
    assert [
        (
            level.level_id,
            level.purpose,
            level.d,
            level.k,
            level.n_obs,
            level.n_int,
            level.noise_var,
            level.budget_slack,
        )
        for level in levels.values()
    ] == [
        (0, "tutorial only", 4, 3, 50, 25, 0.5, 2),
        (1, "small core", 6, 6, 25, 15, 1.0, 1),
        (2, "medium core", 8, 9, 25, 15, 1.0, 1),
        (3, "real scale", 10, 12, 25, 15, 1.0, 1),
        (4, "structural hard", 12, 14, 25, 15, 1.0, 1),
        (5, "noisy hard", 10, 12, 15, 10, 1.5, 0),
    ]


@pytest.mark.parametrize("d", [level.d for level in density_probe_levels().values()])
def test_random_uniform_sampled_edge_count_stays_within_complete_dag_bounds(d):
    max_edges = d * (d - 1) // 2
    counts = [
        len(make_random_uniform_submission(d, np.random.default_rng(seed)).directed_edges)
        for seed in range(100)
    ]

    assert all(0 <= count <= max_edges for count in counts)
