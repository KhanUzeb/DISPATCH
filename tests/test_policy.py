import pytest

from src.router.models import load_model_registry
from src.router.policy import route
from src.router.schemas import ClassificationResult, Provider, RoutingConstraints, Tier


@pytest.fixture
async def registry():
    # Offline: use fallback_models in models.yaml (no live Groq/OpenRouter fetch).
    return await load_model_registry("configs/models.yaml", discover=False)


def _classification(tier: Tier) -> ClassificationResult:
    return ClassificationResult(
        route=tier.value,
        tier=tier,
        similarity_score=0.9,
        passed_threshold=True,
        classifier_version="v2",
        route_config_version="test-routes",
        encoder_id="fake:64",
    )


@pytest.mark.asyncio
async def test_policy_selects_feasible_model(registry):
    decision = route(
        classification=_classification(Tier.CHEAP),
        constraints=RoutingConstraints(),
        registry=registry,
        input_tokens=100,
        policy_version="v2",
        selection_seed="summarize this",
        diversify=True,
    )
    assert decision.selected_model is not None
    assert decision.selected_model.tier == Tier.CHEAP


@pytest.mark.asyncio
async def test_policy_prefers_exact_hard_tier(registry):
    decision = route(
        classification=_classification(Tier.HARD),
        constraints=RoutingConstraints(),
        registry=registry,
        input_tokens=100,
        policy_version="v2",
        selection_seed="design distributed system",
        diversify=True,
    )
    assert decision.selected_model is not None
    assert decision.selected_model.tier == Tier.HARD


@pytest.mark.asyncio
async def test_policy_prefers_fast_models_among_equal_cost(registry):
    """Free-tier diversify must stay within near-latency ties (prefer Groq)."""
    providers: set[str] = set()
    models: set[str] = set()
    for i in range(24):
        decision = route(
            classification=_classification(Tier.CHEAP),
            constraints=RoutingConstraints(),
            registry=registry,
            input_tokens=100,
            policy_version="v2",
            selection_seed=f"cheap-prompt-{i}",
            diversify=True,
        )
        assert decision.selected_model is not None
        providers.add(decision.selected_model.provider.value)
        models.add(decision.selected_model.key)
        # OpenRouter free is much slower in latency guesses — should not win cheap.
        assert decision.selected_model.provider == Provider.GROQ
    assert Provider.GROQ.value in providers
    assert len(models) >= 1


@pytest.mark.asyncio
async def test_policy_never_violates_hard_constraints(registry):
    decision = route(
        classification=_classification(Tier.MID),
        constraints=RoutingConstraints(max_latency_ms=1),
        registry=registry,
        input_tokens=100,
        policy_version="v2",
    )
    assert decision.selected_model is None
    assert decision.error_category is not None


@pytest.mark.asyncio
async def test_policy_enforces_structured_output_capability(registry):
    decision = route(
        classification=_classification(Tier.CHEAP),
        constraints=RoutingConstraints(require_structured_output=True),
        registry=registry,
        input_tokens=100,
        policy_version="v2",
        selection_seed="structured cheap",
    )
    assert decision.selected_model is not None
    assert decision.selected_model.supports_structured_output is True
