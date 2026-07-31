"""build_routing_service wiring and RoutingService.decide_for_client."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from src.router.bootstrap import build_routing_service
from src.router.classifier import Classifier
from src.router.clients import ClientConfig, ClientProfile
from src.router.config import ConfigError, RouterRuntimeConfig, load_config
from src.router.encoder import FakeEncoder
from src.router.executor import Executor
from src.router.index import InMemoryNumpyIndex
from src.router.logging_config import setup_logging
from src.router.models import load_model_registry
from src.router.providers.registry import ProviderRegistry
from src.router.routes import load_route_store
from src.router.schemas import Tier
from src.router.service import ClientRouteDecision, RoutingService


def _offline_env(monkeypatch) -> None:
    monkeypatch.setenv("ROUTER_USE_FAKE_ENCODER", "true")
    monkeypatch.setenv("DISPATCH_DISCOVER", "false")
    monkeypatch.setenv("DISPATCH_PROFILE", "demo")
    monkeypatch.delenv("DISPATCH_MODE", raising=False)


@pytest.fixture
def dispatch_logs(caplog):
    """Capture ``src.*`` logs even though the package logger does not propagate."""
    setup_logging(force=True)
    logging.getLogger("src").propagate = True
    caplog.set_level(logging.INFO, logger="src")
    return caplog


@pytest.mark.asyncio
async def test_build_routing_service_success(monkeypatch, dispatch_logs):
    _offline_env(monkeypatch)
    async with httpx.AsyncClient() as client:
        service, err = await build_routing_service(http_client=client, log_startup=True)
    assert err is None
    assert service is not None
    assert service.active_profile is not None
    assert service.active_profile.mode == "execute"
    assert "fake" in service.classifier.encoder.model_id
    assert "encoder=fake" in dispatch_logs.text


@pytest.mark.asyncio
async def test_build_routing_service_config_error(monkeypatch):
    monkeypatch.setattr(
        "src.router.bootstrap.load_config",
        lambda: (_ for _ in ()).throw(ConfigError("bad router.yaml")),
    )
    service, err = await build_routing_service(log_startup=False)
    assert service is None
    assert err == "bad router.yaml"


# ---------------------------------------------------------------------------
# RoutingService.decide_for_client
# ---------------------------------------------------------------------------


def _runtime_config(**overrides) -> RouterRuntimeConfig:
    base = load_config()
    if not overrides:
        return base
    fields = {**base.__dict__, **overrides}
    return RouterRuntimeConfig(**fields)


async def _service_with_profile(profile: ClientProfile | None) -> RoutingService:
    cfg = _runtime_config()
    store = load_route_store(cfg.routes_path, default_threshold=cfg.default_threshold)
    registry = await load_model_registry(cfg.models_path, discover=False)
    classifier = Classifier(
        route_store=store,
        encoder=FakeEncoder(),
        index=InMemoryNumpyIndex(),
        top_k=cfg.top_k,
        aggregation=cfg.route_aggregation,
    )
    classifier.initialize()
    client_config = None
    if profile is not None:
        client_config = ClientConfig(
            active_profile=profile.name,
            profiles={profile.name: profile},
            path=Path("configs/clients.yaml"),
            version="test",
        )
    return RoutingService(
        classifier,
        Executor(ProviderRegistry(providers={}), max_fallbacks=1),
        registry,
        cfg,
        client_config=client_config,
    )


def _passthrough_profile(**model_overrides: str) -> ClientProfile:
    models = {"cheap": "cheap-m", "mid": "mid-m", "hard": "hard-m", **model_overrides}
    return ClientProfile(
        name="anthropic",
        mode="passthrough",
        protocol="anthropic",
        upstream_base_url="https://api.anthropic.com",
        upstream_api_key="sk-ant-test",
        models=models,
        discover_models=False,
    )


@pytest.mark.asyncio
async def test_decide_for_client_maps_tier_and_caches():
    profile = _passthrough_profile()
    svc = await _service_with_profile(profile)
    client = AsyncMock()

    first = await svc.decide_for_client(
        "summarize this in one sentence",
        client=client,
        profile=profile,
    )
    assert isinstance(first, ClientRouteDecision)
    assert first.cache_hit is False
    assert first.selected_model in profile.models.values()
    assert first.tier in {Tier.CHEAP, Tier.MID, Tier.HARD}
    assert "passthrough:anthropic" in first.reason

    second = await svc.decide_for_client(
        "summarize this in one sentence",
        client=client,
        profile=profile,
    )
    assert second.cache_hit is True
    assert second.selected_model == first.selected_model
    assert second.classification == first.classification


@pytest.mark.asyncio
async def test_decide_for_client_missing_tier_model_raises():
    profile = ClientProfile(
        name="broken",
        mode="passthrough",
        protocol="openai",
        upstream_base_url="https://example.com/v1",
        upstream_api_key="k",
        models={"cheap": "only-cheap", "mid": "", "hard": ""},
        discover_models=False,
    )
    # Bypass ensure_models early-exit by faking has_all_tier_models via pre-filled
    # then clearing mid after construction of a profile that already has all tiers.
    profile = _passthrough_profile()
    svc = await _service_with_profile(profile)
    profile.models["cheap"] = ""
    profile.models["mid"] = ""
    profile.models["hard"] = ""

    async def _noop_ensure(**kwargs):
        return None

    profile.ensure_models = _noop_ensure  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="no model for tier"):
        await svc.decide_for_client("hello world", client=AsyncMock(), profile=profile)
