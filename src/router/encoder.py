from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Sequence


class EncoderError(RuntimeError):
    pass


class Encoder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_id(self) -> str:
        raise NotImplementedError

    @property
    def identity(self) -> str:
        return f"{self.model_id}:{self.dimension}"

    @abstractmethod
    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    def warmup(self) -> None:
        self.encode_queries(["warmup"])


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class FakeEncoder(Encoder):
    """Deterministic hashed n-gram encoder for offline tests (not production quality).

    Prefer HuggingFaceEncoder (MiniLM) for ~real semantic routing. This exists so
    CI/demos work without downloading models.
    """

    def __init__(self, model_id: str = "fake-encoder-v3", dimension: int = 256) -> None:
        self._model_id = model_id
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_id(self) -> str:
        return self._model_id

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            tokens = ["empty"]
        # Unigrams
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + (digest[5] / 255.0)
            vec[idx] += sign * weight
        # Bigrams + character trigrams for short paraphrases
        for left, right in zip(tokens, tokens[1:]):
            digest = hashlib.sha256(f"{left}_{right}".encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += 0.7 * sign
        compact = "".join(tokens)[:80]
        for i in range(max(0, len(compact) - 2)):
            gram = compact[i : i + 3]
            digest = hashlib.sha256(f"c3:{gram}".encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += 0.25 * sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]


class HuggingFaceEncoder(Encoder):
    def __init__(self, model_id: str, device: str | None = None) -> None:
        self._model_id = model_id
        self._device = device
        self._backend = None
        self._dimension: int | None = None

    def _ensure_backend(self):
        if self._backend is not None:
            return self._backend
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise EncoderError("sentence-transformers is required for HuggingFaceEncoder") from exc
        self._backend = SentenceTransformer(self._model_id, device=self._device)
        vector = self._backend.encode(["dimension-check"])[0]
        self._dimension = len(vector)
        return self._backend

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._ensure_backend()
        return int(self._dimension or 0)

    @property
    def model_id(self) -> str:
        return self._model_id

    def _validate_vectors(self, vectors: list[list[float]]) -> None:
        expected = self.dimension
        for v in vectors:
            if len(v) != expected:
                raise EncoderError(f"Unexpected embedding dimension {len(v)} != {expected}")

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        backend = self._ensure_backend()
        vectors = backend.encode(list(texts)).tolist()
        self._validate_vectors(vectors)
        return vectors

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        backend = self._ensure_backend()
        vectors = backend.encode(list(texts)).tolist()
        self._validate_vectors(vectors)
        return vectors
