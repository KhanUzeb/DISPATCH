from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .schemas import RouteDefinition


@dataclass(frozen=True)
class IndexHit:
    route_name: str
    utterance: str
    similarity_score: float


class SemanticIndex(ABC):
    @abstractmethod
    def build_or_load(self, routes: Sequence[RouteDefinition], embeddings: Sequence[Sequence[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, query_embedding: Sequence[float], top_k: int) -> list[IndexHit]:
        raise NotImplementedError


class InMemoryNumpyIndex(SemanticIndex):
    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._rows: list[tuple[str, str]] = []

    def build_or_load(self, routes: Sequence[RouteDefinition], embeddings: Sequence[Sequence[float]]) -> None:
        rows: list[tuple[str, str]] = []
        for route in routes:
            for utterance in route.utterances:
                rows.append((route.name, utterance))
        if not rows:
            self._vectors = np.zeros((0, 1), dtype=np.float32)
            self._rows = []
            return
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix")
        if len(rows) != len(matrix):
            raise ValueError("Number of route utterances and embeddings must match")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vectors = matrix / norms
        self._rows = rows

    def query(self, query_embedding: Sequence[float], top_k: int) -> list[IndexHit]:
        if self._vectors is None or len(self._rows) == 0:
            return []
        query_vec = np.array(query_embedding, dtype=np.float32)
        if query_vec.ndim != 1 or query_vec.size == 0:
            return []
        if self._vectors.shape[1] != query_vec.shape[0]:
            return []
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        normalized_query = query_vec / query_norm
        scores = self._vectors @ normalized_query
        k = max(1, min(int(top_k), len(scores)))
        top_indices = np.argsort(scores)[::-1][:k]
        return [
            IndexHit(
                route_name=self._rows[i][0],
                utterance=self._rows[i][1],
                similarity_score=float(scores[i]),
            )
            for i in top_indices
        ]
