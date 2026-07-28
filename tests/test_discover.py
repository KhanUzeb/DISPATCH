"""Unit tests for live model discovery + tier assignment."""

from __future__ import annotations

from src.router.discover import (
    DiscoveredModel,
    TieringConfig,
    assign_tier,
    bucket_and_limit,
    estimate_params_b,
    pick_tier_representatives,
)
from src.router.models import load_model_registry
from src.router.schemas import Provider, Tier


def test_estimate_params_from_model_id():
    assert estimate_params_b("llama-3.1-8b-instant") == 8.0
    assert estimate_params_b("meta-llama/llama-3.3-70b-instruct:free") == 70.0
    assert estimate_params_b("openai/gpt-oss-120b") == 120.0


def test_assign_tier_by_size_and_hints():
    tiering = TieringConfig()
    cheap = DiscoveredModel("llama-3.1-8b-instant", Provider.GROQ, name="Llama 8B")
    mid = DiscoveredModel("llama-3.3-70b-versatile", Provider.GROQ, name="Llama 70B")
    hard = DiscoveredModel("openai/gpt-oss-120b", Provider.GROQ, name="120B")
    assert assign_tier(cheap, tiering) == Tier.CHEAP
    assert assign_tier(mid, tiering) == Tier.MID
    assert assign_tier(hard, tiering) == Tier.HARD
    assert assign_tier(DiscoveredModel("claude-3-5-haiku-latest", Provider.ANTHROPIC), tiering) == Tier.CHEAP
    assert assign_tier(DiscoveredModel("claude-opus-4", Provider.ANTHROPIC), tiering) == Tier.HARD
    assert assign_tier(DiscoveredModel("qwen/qwen3.6-27b", Provider.GROQ), tiering) == Tier.HARD


def test_is_chat_model_filters_music_and_embeddings():
    from src.router.discover import _is_chat_model

    assert _is_chat_model("google/lyria-3-clip-preview", {"architecture": {"modality": "text->audio"}}) is False
    assert _is_chat_model("openrouter/free", {}) is False
    assert _is_chat_model("poolside/laguna-m.1:free", {}) is False
    assert _is_chat_model(
        "meta-llama/llama-3.3-70b-instruct:free",
        {"architecture": {"modality": "text->text", "output_modalities": ["text"]}},
    )
    assert _is_chat_model("whisper-large-v3", {}) is False


def test_bucket_caps_openrouter_when_groq_available():
    models = [
        DiscoveredModel("llama-3.1-8b-instant", Provider.GROQ, name="8b"),
        DiscoveredModel("meta-llama/llama-3.2-3b-instruct:free", Provider.OPENROUTER, name="3b"),
        DiscoveredModel("qwen/qwen3-4b:free", Provider.OPENROUTER, name="4b"),
        DiscoveredModel("google/gemma-2-9b-it:free", Provider.OPENROUTER, name="9b"),
        DiscoveredModel("mistralai/mistral-7b-instruct:free", Provider.OPENROUTER, name="7b"),
    ]
    specs = bucket_and_limit(models, TieringConfig(max_per_tier=4, prefer_free=True))
    cheap = [s for s in specs if s.tier == Tier.CHEAP]
    assert any(s.provider == Provider.GROQ for s in cheap)
    assert sum(1 for s in cheap if s.provider == Provider.OPENROUTER) <= 1


def test_bucket_prefer_free_and_limit():
    models = [
        DiscoveredModel("paid-big", Provider.OPENROUTER, cost_per_1k_input=1.0, cost_per_1k_output=1.0, name="405b"),
        DiscoveredModel("free-small", Provider.OPENROUTER, name="3b"),
        DiscoveredModel("free-mid", Provider.GROQ, name="70b"),
        DiscoveredModel("free-hard", Provider.GROQ, name="120b"),
    ]
    specs = bucket_and_limit(models, TieringConfig(max_per_tier=2, prefer_free=True))
    ids = {s.provider_model_id for s in specs}
    assert "free-small" in ids
    assert "paid-big" not in ids  # free preferred within its would-be tier


def test_pick_tier_representatives():
    models = [
        DiscoveredModel("llama-3.1-8b-instant", Provider.GROQ, name="8b"),
        DiscoveredModel("llama-3.3-70b-versatile", Provider.GROQ, name="70b"),
        DiscoveredModel("openai/gpt-oss-120b", Provider.GROQ, name="120b"),
    ]
    specs = bucket_and_limit(models, TieringConfig())
    picked = pick_tier_representatives(specs)
    assert set(picked) == {"cheap", "mid", "hard"}


def test_fallback_registry_loads_without_network(monkeypatch):
    monkeypatch.setenv("DISPATCH_DISCOVER", "false")
    registry = load_model_registry("configs/models.yaml", discover=False)
    assert registry.source in {"static", "fallback"}
    assert registry.get_by_tier(Tier.CHEAP)
    assert registry.get_by_tier(Tier.MID)
    assert registry.get_by_tier(Tier.HARD)
    providers = {m.provider for m in registry.enabled_models()}
    assert Provider.GROQ in providers
    assert Provider.OPENROUTER in providers
