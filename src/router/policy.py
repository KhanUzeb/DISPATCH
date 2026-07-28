from __future__ import annotations

import hashlib

from .models import ModelRegistry
from .schemas import ClassificationResult, ModelSpec, RoutingCandidate, RoutingConstraints, RoutingDecision, RoutingErrorCategory, Tier


def _estimated_cost(model: ModelSpec, input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens / 1000) * model.cost_per_1k_input
        + (output_tokens / 1000) * model.cost_per_1k_output
    )


def _hard_rejections(model: ModelSpec, constraints: RoutingConstraints, input_tokens: int) -> list[str]:
    reasons: list[str] = []
    if constraints.min_context_window and model.context_window < constraints.min_context_window:
        reasons.append("context_window")
    if constraints.max_latency_ms and model.avg_latency_ms > constraints.max_latency_ms:
        reasons.append("latency")
    if constraints.allow_providers and model.provider not in constraints.allow_providers:
        reasons.append("provider_allow_list")
    if constraints.deny_providers and model.provider in constraints.deny_providers:
        reasons.append("provider_deny_list")
    if constraints.require_structured_output and not model.supports_structured_output:
        reasons.append("structured_output")
    if constraints.require_tool_calling and not model.supports_tool_calling:
        reasons.append("tool_calling")
    if constraints.explicit_model_allowlist and model.key not in constraints.explicit_model_allowlist:
        reasons.append("model_allowlist")
    if constraints.max_cost_usd is not None:
        estimated = _estimated_cost(model, input_tokens, constraints.estimated_output_tokens)
        if estimated > constraints.max_cost_usd:
            reasons.append("max_cost")
    return reasons


def _tier_rank(tier: Tier) -> int:
    return {Tier.CHEAP: 0, Tier.MID: 1, Tier.HARD: 2}[tier]


def _sticky_index(seed: str, size: int) -> int:
    if size <= 1:
        return 0
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % size


def _select_from_pool(
    pool: list[RoutingCandidate],
    objective: str,
    *,
    diversify: bool,
    selection_seed: str,
) -> RoutingCandidate:
    if len(pool) == 1:
        return pool[0]

    if objective == "lowest_latency":
        # Still diversify among near-latency ties when requested.
        best_latency = min(c.model.avg_latency_ms for c in pool)
        near = [c for c in pool if c.model.avg_latency_ms <= best_latency * 1.15]
        ranked = sorted(near, key=lambda c: (c.model.avg_latency_ms, c.model.key))
        if diversify:
            return ranked[_sticky_index(selection_seed, len(ranked))]
        return ranked[0]

    # lowest_cost (default): among equal-cost models (typical free demo pool),
    # prefer low latency so we don't sticky-pick a slow OpenRouter free endpoint.
    # Diversify only among near-latency ties (usually fast Groq ids).
    min_cost = min(c.estimated_cost_usd or 0.0 for c in pool)
    cheapest = [c for c in pool if abs((c.estimated_cost_usd or 0.0) - min_cost) < 1e-12]
    if not diversify:
        return min(cheapest, key=lambda c: (c.model.avg_latency_ms, c.model.key))

    best_latency = min(c.model.avg_latency_ms for c in cheapest)
    near = [c for c in cheapest if c.model.avg_latency_ms <= best_latency * 1.25]
    ranked = sorted(near, key=lambda c: (c.model.avg_latency_ms, c.model.key))
    return ranked[_sticky_index(selection_seed + ":model", len(ranked))]


def route(
    classification: ClassificationResult,
    constraints: RoutingConstraints,
    registry: ModelRegistry,
    input_tokens: int,
    policy_version: str,
    *,
    selection_seed: str | None = None,
    diversify: bool = True,
) -> RoutingDecision:
    # Prefer exact classified tier first. Only upgrade when that tier has no
    # feasible model, never silently downgrade below the classified tier.
    tier_order = [classification.tier]
    if classification.tier == Tier.CHEAP:
        tier_order += [Tier.MID, Tier.HARD]
    elif classification.tier == Tier.MID:
        tier_order += [Tier.HARD]

    candidate_models: list[ModelSpec] = []
    for tier in tier_order:
        candidate_models.extend(registry.get_by_tier(tier))

    diagnostics: list[RoutingCandidate] = []
    feasible: list[RoutingCandidate] = []
    for model in candidate_models:
        estimated_cost = _estimated_cost(model, input_tokens, constraints.estimated_output_tokens)
        rejected = _hard_rejections(model, constraints, input_tokens)
        candidate = RoutingCandidate(
            model=model,
            rejected_reasons=rejected,
            estimated_cost_usd=estimated_cost,
            objective_score=None if rejected else estimated_cost,
        )
        diagnostics.append(candidate)
        if candidate.is_feasible:
            feasible.append(candidate)

    if not feasible:
        return RoutingDecision(
            selected_model=None,
            classification=classification,
            candidates=diagnostics,
            reason="no feasible model satisfying hard constraints",
            policy_version=policy_version,
            error_category=RoutingErrorCategory.NO_FEASIBLE_MODEL,
        )

    required = _tier_rank(classification.tier)
    same_or_above = [c for c in feasible if _tier_rank(c.model.tier) >= required]
    pool = same_or_above or feasible

    exact = [c for c in pool if c.model.tier == classification.tier]
    if exact:
        pool = exact

    objective = constraints.optimization_objective or "lowest_cost"
    seed = selection_seed or classification.route or classification.tier.value
    best = _select_from_pool(pool, objective, diversify=diversify, selection_seed=seed)

    providers_in_pool = sorted({c.model.provider.value for c in pool})
    return RoutingDecision(
        selected_model=best.model,
        classification=classification,
        candidates=diagnostics,
        reason=(
            f"selected by policy objective={objective} tier={classification.tier.value} "
            f"diversify={diversify} pool_providers={providers_in_pool}"
        ),
        policy_version=policy_version,
    )
