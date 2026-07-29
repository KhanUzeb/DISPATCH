from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .schemas import RouteDefinition, Tier


class RouteConfigError(RuntimeError):
    pass


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "to",
        "of",
        "in",
        "on",
        "for",
        "and",
        "or",
        "is",
        "are",
        "be",
        "as",
        "at",
        "by",
        "with",
        "from",
        "it",
        "its",
        "into",
        "me",
        "my",
        "your",
        "please",
        "how",
        "what",
        "when",
        "where",
        "which",
        "who",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "about",
        "using",
        "use",
        "used",
        "more",
        "most",
        "than",
        "then",
        "them",
        "they",
        "we",
        "you",
        "i",
    }
)


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass(frozen=True)
class ScoringConfig:
    accept_floor: float = 0.32
    margin: float = 0.04
    overlap_weight: float = 0.18
    overlap_cap: float = 0.22
    centroid_weight: float = 0.35
    # Prefer cheaper tier when top route scores fall within this band.
    ambiguity_band: float = 0.08
    # Do not apply cheap-bias when confidence is already high.
    ambiguity_max_score: float = 0.56


@dataclass(frozen=True)
class RouteStore:
    routes: list[RouteDefinition]
    version: str
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    # Auto-mined from exemplars: tier -> token -> weight in [0, 1]
    keyword_weights: dict[str, dict[str, float]] = field(default_factory=dict)

    def by_name(self, route_name: str) -> RouteDefinition | None:
        return next((r for r in self.routes if r.name == route_name), None)

    def overlap_scores(self, prompt: str) -> dict[str, float]:
        """Soft boost from exemplar keyword overlap — no hand-written regex."""
        prompt_tokens = set(_tokens(prompt))
        if not prompt_tokens:
            return {route.name: 0.0 for route in self.routes}
        scores: dict[str, float] = {}
        for route in self.routes:
            weights = self.keyword_weights.get(route.name, {})
            if not weights:
                scores[route.name] = 0.0
                continue
            hit = sum(weights[t] for t in prompt_tokens if t in weights)
            denom = math.sqrt(len(prompt_tokens)) * math.sqrt(sum(weights.values()) or 1.0)
            raw = hit / denom if denom else 0.0
            scores[route.name] = min(1.0, raw)
        return scores


def _normalize_utterances(utterances: list[str]) -> list[str]:
    normalized = [u.strip() for u in utterances if isinstance(u, str) and u.strip()]
    if not normalized:
        raise RouteConfigError("Route utterances cannot be empty")
    return normalized


def _mine_keyword_weights(routes: list[RouteDefinition]) -> dict[str, dict[str, float]]:
    """TF-IDF-like weights from exemplars only (config-driven, not hard-coded cues)."""
    doc_freqs: Counter[str] = Counter()
    per_route: dict[str, Counter[str]] = {}
    for route in routes:
        counts: Counter[str] = Counter()
        for utterance in route.utterances:
            toks = set(_tokens(utterance))
            counts.update(toks)
        per_route[route.name] = counts
        doc_freqs.update(counts.keys())

    n_routes = max(1, len(routes))
    weights: dict[str, dict[str, float]] = {}
    for route in routes:
        counts = per_route[route.name]
        if not counts:
            weights[route.name] = {}
            continue
        max_tf = max(counts.values())
        route_weights: dict[str, float] = {}
        for token, tf in counts.items():
            idf = math.log((1 + n_routes) / (1 + doc_freqs[token])) + 1.0
            route_weights[token] = (tf / max_tf) * idf
        # Normalize to unit sum for stable boosts.
        total = sum(route_weights.values()) or 1.0
        weights[route.name] = {t: w / total for t, w in route_weights.items()}
    return weights


def _load_scoring(raw: dict) -> ScoringConfig:
    # Prefer new `scoring` block; accept legacy `lexical` knobs if present.
    section = raw.get("scoring") or raw.get("lexical") or {}
    if not isinstance(section, dict):
        raise RouteConfigError("configs/routes.yaml: scoring must be a mapping")
    return ScoringConfig(
        accept_floor=float(section.get("accept_floor", 0.32)),
        margin=float(section.get("margin", 0.04)),
        overlap_weight=float(section.get("overlap_weight", section.get("boost_per_hit", 0.18))),
        overlap_cap=float(section.get("overlap_cap", section.get("boost_cap", 0.22))),
        centroid_weight=float(section.get("centroid_weight", 0.35)),
        ambiguity_band=float(section.get("ambiguity_band", 0.08)),
        ambiguity_max_score=float(section.get("ambiguity_max_score", 0.56)),
    )


def load_route_store(path: str | Path, default_threshold: float = 0.7) -> RouteStore:
    route_path = Path(path)
    raw = yaml.safe_load(route_path.read_text(encoding="utf-8")) or {}
    tiers = raw.get("tiers", {})
    thresholds = raw.get("thresholds", {})
    if not isinstance(tiers, dict):
        raise RouteConfigError("configs/routes.yaml: tiers must be a mapping")

    seen_names: set[str] = set()
    routes: list[RouteDefinition] = []
    for tier_name, utterances in tiers.items():
        if tier_name not in {t.value for t in Tier}:
            raise RouteConfigError(f"Unknown tier in routes config: {tier_name}")
        normalized = _normalize_utterances(utterances or [])
        route_name = tier_name
        if route_name in seen_names:
            raise RouteConfigError(f"Duplicate route name: {route_name}")
        seen_names.add(route_name)
        threshold = float(thresholds.get(route_name, default_threshold))
        if not 0.0 <= threshold <= 1.0:
            raise RouteConfigError(f"Invalid threshold for route {route_name}: {threshold}")
        routes.append(
            RouteDefinition(
                name=route_name,
                tier=Tier(route_name),
                utterances=normalized,
                threshold=threshold,
            )
        )

    if not routes:
        raise RouteConfigError("No routes configured")

    scoring = _load_scoring(raw)
    keyword_weights = _mine_keyword_weights(routes)

    normalized_dump = yaml.safe_dump(
        {
            "routes": [
                {
                    "name": route.name,
                    "tier": route.tier.value,
                    "threshold": route.threshold,
                    "utterances": route.utterances,
                }
                for route in sorted(routes, key=lambda r: r.name)
            ],
            "scoring": {
                "accept_floor": scoring.accept_floor,
                "margin": scoring.margin,
                "overlap_weight": scoring.overlap_weight,
                "overlap_cap": scoring.overlap_cap,
                "centroid_weight": scoring.centroid_weight,
                "ambiguity_band": scoring.ambiguity_band,
                "ambiguity_max_score": scoring.ambiguity_max_score,
            },
        },
        sort_keys=True,
    )
    version = hashlib.sha256(normalized_dump.encode("utf-8")).hexdigest()[:16]
    return RouteStore(
        routes=routes,
        version=version,
        scoring=scoring,
        keyword_weights=keyword_weights,
    )
