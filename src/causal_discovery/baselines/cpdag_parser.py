"""Shared parsing for causal-learn endpoint matrices."""

from __future__ import annotations

import numpy as np

from causal_discovery.scoring.submission import DirectedEdge, UndirectedEdge


def parse_causallearn_endpoint_matrix(
    endpoint_matrix: np.ndarray,
) -> tuple[frozenset[DirectedEdge], frozenset[UndirectedEdge]]:
    """Parse a causal-learn endpoint matrix into directed/undirected edges.

    Supported endpoint patterns for pair (i, j):
    - (-1, 1): i -> j
    - (1, -1): j -> i
    - (-1, -1): i -- j
    - (0, 0): no edge
    """
    matrix = np.asarray(endpoint_matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "endpoint_matrix must be square, got shape "
            f"{tuple(int(x) for x in matrix.shape)}"
        )

    d = int(matrix.shape[0])
    directed: set[DirectedEdge] = set()
    undirected: set[UndirectedEdge] = set()

    for i in range(d):
        for j in range(i + 1, d):
            pair = (int(matrix[i, j]), int(matrix[j, i]))
            if pair == (-1, 1):
                directed.add((i, j))
            elif pair == (1, -1):
                directed.add((j, i))
            elif pair == (-1, -1):
                undirected.add((i, j))
            elif pair == (0, 0):
                continue
            else:
                raise ValueError(
                    f"Unsupported causal-learn endpoint state for pair {(i, j)}: {pair}"
                )

    return frozenset(directed), frozenset(undirected)
