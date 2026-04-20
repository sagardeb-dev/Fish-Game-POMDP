"""Node relabeling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class Permutation:
    """A bijection over node ids 0..d-1."""

    mapping: tuple[int, ...]

    def __post_init__(self) -> None:
        normalized = tuple(int(value) for value in self.mapping)
        object.__setattr__(self, "mapping", normalized)
        expected = set(range(len(self.mapping)))
        if set(self.mapping) != expected:
            raise ValueError(
                "Permutation mapping must be a bijection over 0..d-1, "
                f"got {self.mapping}"
            )

    @classmethod
    def from_mapping(cls, mapping: Iterable[int]) -> "Permutation":
        return cls(mapping=tuple(mapping))

    @property
    def size(self) -> int:
        return len(self.mapping)

    def apply_node(self, node: int) -> int:
        if node < 0 or node >= self.size:
            raise ValueError(f"Node {node} out of range for permutation of size {self.size}")
        return self.mapping[node]

    def apply_edges(self, edges: Iterable[tuple[int, int]]) -> frozenset[tuple[int, int]]:
        return frozenset((self.apply_node(src), self.apply_node(dst)) for src, dst in edges)

    def apply_square_matrix(self, matrix: np.ndarray) -> np.ndarray:
        array = np.array(matrix, dtype=float, copy=True)
        if array.shape != (self.size, self.size):
            raise ValueError(
                f"Matrix shape {array.shape} does not match permutation size {self.size}"
            )
        result = np.zeros_like(array)
        for old_i, new_i in enumerate(self.mapping):
            for old_j, new_j in enumerate(self.mapping):
                result[new_i, new_j] = array[old_i, old_j]
        return result

    def apply_vector(self, vector: np.ndarray) -> np.ndarray:
        array = np.array(vector, dtype=float, copy=True)
        if array.shape != (self.size,):
            raise ValueError(
                f"Vector shape {array.shape} does not match permutation size {self.size}"
            )
        result = np.zeros_like(array)
        for old_i, new_i in enumerate(self.mapping):
            result[new_i] = array[old_i]
        return result
