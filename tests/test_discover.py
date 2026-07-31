"""Unit tests for live model discovery + tier assignment."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.router.discover import (
    DiscoveredModel,
    TieringConfig,
    assign_tier,
    bucket_and_limit,
    estimate_params_b,
    fetch_openai_compatible_models,
    fetch_openrouter_models,
    pick_tier_representatives,
)
from src.router.models import load_model_registry
from src.router.schemas import Provider, Tier


def test_pick_tier_representatives():
    models = [
        DiscoveredModel("llama-3.1-8b-instant", Provider.GROQ, name="8b"),
        DiscoveredModel("llama-3.3-70b-versatile", Provider.GROQ, name="70b"),
        DiscoveredModel("openai/gpt-oss-120b", Provider.GROQ, name="120b"),
    ]
    specs = bucket_and_limit(models, TieringConfig())
    picked = pick_tier_representatives(specs)
    assert set(picked) == {"cheap", "mid", "hard"}


@pytest.mark.asyncio
async def test_fallback_registry_loads_without_network(monkeypatch):
    monkeypatch.setenv("DISPATCH_DISCOVER", "false")
    registry = await load_model_registry("configs/models.yaml", discover=False)
    assert registry.source in {"static", "fallback"}
    assert registry.get_by_tier(Tier.CHEAP)
    assert registry.get_by_tier(Tier.MID)
    assert registry.get_by_tier(Tier.HARD)
    providers = {m.provider for m in registry.enabled_models()}
    assert Provider.GROQ in providers
    assert Provider.OPENROUTER in providers


