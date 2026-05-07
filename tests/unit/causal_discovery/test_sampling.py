import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from causal_discovery import DAG, LinearGaussianSCM, sample_interventional_data, sample_observational_data


def _make_chain_scm() -> LinearGaussianSCM:
    dag = DAG.from_edges(3, [(0, 1), (1, 2)])
    return LinearGaussianSCM.from_dag(
        dag,
        weights=np.array(
            [
                [0.0, 0.8, 0.0],
                [0.0, 0.0, 0.7],
                [0.0, 0.0, 0.0],
            ]
        ),
        noise_var=1.0,
    )


def test_sample_observational_data_shape_and_finiteness():
    scm = _make_chain_scm()
    data = sample_observational_data(scm, n_samples=50, rng=np.random.default_rng(7))
    assert data.shape == (50, 3)
    assert np.isfinite(data).all()


def test_sample_interventional_data_shape_and_finiteness():
    scm = _make_chain_scm()
    data = sample_interventional_data(
        scm,
        var=1,
        value=0.0,
        n_samples=80,
        rng=np.random.default_rng(7),
    )
    assert data.shape == (80, 3)
    assert np.isfinite(data).all()


def test_intervention_sets_intervened_column_exactly():
    scm = _make_chain_scm()
    data = sample_interventional_data(
        scm,
        var=1,
        value=3.25,
        n_samples=80,
        rng=np.random.default_rng(42),
    )
    assert np.all(data[:, 1] == 3.25)


def test_downstream_nodes_remain_stochastic_under_intervention():
    scm = _make_chain_scm()
    data = sample_interventional_data(
        scm,
        var=1,
        value=0.0,
        n_samples=200,
        rng=np.random.default_rng(42),
    )
    assert np.var(data[:, 2]) > 1e-6


def test_invalid_intervention_variable_raises():
    scm = _make_chain_scm()
    with pytest.raises(ValueError, match="out of range"):
        sample_interventional_data(
            scm,
            var=7,
            value=0.0,
            n_samples=10,
            rng=np.random.default_rng(1),
        )


def test_sampling_is_reproducible_for_fixed_seed():
    scm = _make_chain_scm()
    obs1 = sample_observational_data(scm, 40, np.random.default_rng(123))
    obs2 = sample_observational_data(scm, 40, np.random.default_rng(123))
    int1 = sample_interventional_data(scm, 1, 0.0, 40, np.random.default_rng(123))
    int2 = sample_interventional_data(scm, 1, 0.0, 40, np.random.default_rng(123))
    assert np.array_equal(obs1, obs2)
    assert np.array_equal(int1, int2)

