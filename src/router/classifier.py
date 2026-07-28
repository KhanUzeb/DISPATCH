"""Semantic classifier — embeddings + exemplar overlap (no hand-written regex).

Primary signal: cosine similarity to route exemplars (and tier centroids).
Secondary soft signal: keyword overlap auto-mined from the same exemplars in
configs/routes.yaml. Unmatched prompts fall back to mid.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .encoder import Encoder, EncoderError
from .index import SemanticIndex
from .routes import RouteStore
from .schemas import ClassificationResult, Tier

DEFAULT_TIER = Tier.MID
CLASSIFIER_VERSION = "v4"
_TIER_RANK = {Tier.CHEAP: 0, Tier.MID: 1, Tier.HARD: 2}


def _cosine(a: list[float], b: list[float]) -> float:
    # Vectors from FakeEncoder / ST are L2-normalized; still guard.
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


@dataclass
class Classifier:
    route_store: RouteStore
    encoder: Encoder
    index: SemanticIndex
    top_k: int = 8
    aggregation: str = "max"
    accept_floor: float | None = None
    margin: float | None = None
    _centroids: dict[str, list[float]] = field(default_factory=dict, init=False, repr=False)
    _utterance_route: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        scoring = self.route_store.scoring
        if self.accept_floor is None:
            self.accept_floor = scoring.accept_floor
        if self.margin is None:
            self.margin = scoring.margin

    def initialize(self) -> None:
        utterances: list[str] = []
        route_names: list[str] = []
        for route in self.route_store.routes:
            for utterance in route.utterances:
                utterances.append(utterance)
                route_names.append(route.name)
        doc_embeddings = self.encoder.encode_documents(utterances)
        self.index.build_or_load(self.route_store.routes, doc_embeddings)
        self._utterance_route = route_names
        self._centroids = self._build_centroids(route_names, doc_embeddings)

    def _build_centroids(
        self,
        route_names: list[str],
        embeddings: list[list[float]],
    ) -> dict[str, list[float]]:
        buckets: dict[str, list[list[float]]] = defaultdict(list)
        for name, vec in zip(route_names, embeddings):
            buckets[name].append(vec)
        centroids: dict[str, list[float]] = {}
        for name, vectors in buckets.items():
            dim = len(vectors[0])
            acc = [0.0] * dim
            for vec in vectors:
                for i, v in enumerate(vec):
                    acc[i] += v
            n = float(len(vectors))
            mean = [v / n for v in acc]
            norm = sum(v * v for v in mean) ** 0.5 or 1.0
            centroids[name] = [v / norm for v in mean]
        return centroids

    def _classification_text(self, prompt: str) -> str:
        """Prefer the instruction head so long payloads do not dilute the task."""
        text = prompt.strip()
        if len(text) <= 180:
            return text
        # Keep the leading instruction (before a long colon/newline payload).
        for sep in (":\n", ":\r\n", "\n\n"):
            if sep in text:
                head, tail = text.split(sep, 1)
                if 12 <= len(head) <= 160 and len(tail) > 40:
                    return head.strip()
        if ": " in text[:160]:
            head, tail = text.split(": ", 1)
            if 12 <= len(head) <= 120 and len(tail) > 40:
                return f"{head.strip()}: {tail[:80].strip()}"
        return text[:180]

    def classify(self, prompt: str, expects_structured_output: bool = False) -> ClassificationResult:
        normalized = prompt.strip()
        if not normalized:
            return self._fallback("blank prompt")
        query_text = self._classification_text(normalized)
        try:
            query_embedding = self.encoder.encode_queries([query_text])[0]
            hits = self.index.query(query_embedding, self.top_k)
        except (EncoderError, ValueError):
            return self._fallback("encoder/index error")

        if not hits:
            return self._fallback("no route matches")

        grouped: dict[str, list[float]] = defaultdict(list)
        for hit in hits:
            grouped[hit.route_name].append(hit.similarity_score)

        semantic_scores: dict[str, float] = {}
        for route_name, scores in grouped.items():
            if self.aggregation == "mean":
                semantic_scores[route_name] = sum(scores) / len(scores)
            else:
                semantic_scores[route_name] = max(scores)

        scoring = self.route_store.scoring
        # Blend in tier centroid similarity (more stable than a single utterance).
        centroid_weight = max(0.0, min(1.0, scoring.centroid_weight))
        for route in self.route_store.routes:
            centroid = self._centroids.get(route.name)
            if centroid is None:
                continue
            c_sim = _cosine(query_embedding, centroid)
            base = semantic_scores.get(route.name, 0.0)
            semantic_scores[route.name] = (1.0 - centroid_weight) * base + centroid_weight * c_sim

        # Soft exemplar-keyword overlap (auto-mined — not regex lists).
        overlap = self.route_store.overlap_scores(query_text)
        route_scores: dict[str, float] = {}
        for route in self.route_store.routes:
            base = semantic_scores.get(route.name, 0.0)
            boost = min(scoring.overlap_cap, overlap.get(route.name, 0.0) * scoring.overlap_weight)
            route_scores[route.name] = min(1.0, base + boost)

        if not route_scores:
            return self._fallback("no route scores")

        ranked = sorted(route_scores.items(), key=lambda item: item[1], reverse=True)
        best_score = ranked[0][1]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        ambiguity_band = max(0.0, float(scoring.ambiguity_band))
        near = [(name, score) for name, score in ranked if best_score - score <= ambiguity_band]
        # Ambiguous matches → cheaper tier (demo-friendly, cost-aware).
        if len(near) > 1:
            near_sorted = sorted(
                near,
                key=lambda item: (
                    _TIER_RANK.get(self.route_store.by_name(item[0]).tier, 99)
                    if self.route_store.by_name(item[0]) is not None
                    else 99,
                    -item[1],
                ),
            )
            route_name, best_score = near_sorted[0]
            accepted_via = "ambiguity_prefer_cheap"
        else:
            route_name = ranked[0][0]
            accepted_via = "top_score"

        route = self.route_store.by_name(route_name)
        if route is None:
            return self._fallback("route not found in store")

        accept_floor = float(self.accept_floor if self.accept_floor is not None else 0.32)
        margin = float(self.margin if self.margin is not None else 0.04)
        cleared_threshold = best_score >= route.threshold
        clear_winner = (best_score - second_score) >= margin and best_score >= accept_floor
        passed_threshold = cleared_threshold or clear_winner or accepted_via == "ambiguity_prefer_cheap"

        if not passed_threshold:
            return self._fallback(
                "no route cleared threshold",
                best_score=best_score,
                diagnostics={
                    "route_scores": route_scores,
                    "semantic_scores": semantic_scores,
                    "overlap_scores": overlap,
                    "second_score": second_score,
                    "query_text": query_text,
                },
            )

        tier = route.tier
        if expects_structured_output and tier == Tier.CHEAP:
            tier = Tier.MID

        return ClassificationResult(
            route=route.name,
            tier=tier,
            similarity_score=float(best_score),
            passed_threshold=True,
            classifier_version=CLASSIFIER_VERSION,
            route_config_version=self.route_store.version,
            encoder_id=self.encoder.identity,
            diagnostics={
                "route_scores": route_scores,
                "semantic_scores": semantic_scores,
                "overlap_scores": overlap,
                "accepted_via": (
                    "threshold"
                    if cleared_threshold and accepted_via != "ambiguity_prefer_cheap"
                    else ("margin" if clear_winner and accepted_via != "ambiguity_prefer_cheap" else accepted_via)
                ),
                "query_text": query_text,
            },
        )

    def _fallback(
        self,
        reason: str,
        best_score: float = 0.0,
        diagnostics: dict | None = None,
    ) -> ClassificationResult:
        return ClassificationResult(
            route=None,
            tier=DEFAULT_TIER,
            similarity_score=best_score,
            passed_threshold=False,
            classifier_version=CLASSIFIER_VERSION,
            route_config_version=self.route_store.version,
            encoder_id=self.encoder.identity,
            fallback_reason=reason,
            diagnostics=diagnostics or {},
        )
