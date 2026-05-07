import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from causal_discovery.baselines import parse_causallearn_endpoint_matrix


def test_parse_causallearn_endpoint_matrix_supported_states():
    matrix = np.array(
        [
            [0, -1, 0, 0],
            [1, 0, -1, 1],
            [0, -1, 0, -1],
            [0, -1, -1, 0],
        ],
        dtype=int,
    )
    directed, undirected = parse_causallearn_endpoint_matrix(matrix)
    assert directed == frozenset({(0, 1), (3, 1)})
    assert undirected == frozenset({(1, 2), (2, 3)})


def test_parse_causallearn_endpoint_matrix_rejects_unsupported_pair():
    matrix = np.array([[0, 1], [1, 0]], dtype=int)
    with pytest.raises(ValueError, match="Unsupported causal-learn endpoint state"):
        parse_causallearn_endpoint_matrix(matrix)
