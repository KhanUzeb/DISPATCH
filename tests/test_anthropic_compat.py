"""Anthropic /v1/messages shim — end-to-end handler branches."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.anthropic_compat import (
    AnthropicMessage,
    MessagesRequest,
    anthropic_error,
    anthropic_messages_to_prompt,
)
from src.api import app
from src.router.clients import ClientProfile
from src.router.passthrough import PassthroughResult
from src.router.schemas import ClassificationResult, Tier
from src.router.security import SlidingWindowRateLimiter, require_router_auth
from src.router.service import ClientRouteDecision


def _classification(tier: Tier = Tier.CHEAP) -> ClassificationResult:
    return ClassificationResult(
        route=tier.value,
        tier=tier,
        similarity_score=0.88,
        passed_threshold=True,
        classifier_version="c1",
        route_config_version="r1",
        encoder_id="fake:64",
    )


def _anthropic_profile(**overrides) -> ClientProfile:
    data = dict(
        name="anthropic",
        mode="passthrough",
        protocol="anthropic",
        upstream_base_url="https://api.anthropic.com",
        upstream_api_key="sk-ant-test",
        models={"cheap": "claude-haiku", "mid": "claude-sonnet", "hard": "claude-opus"},
        discover_models=False,
    )
    data.update(overrides)
    return ClientProfile(**data)


def _svc(profile: ClientProfile | None, **methods) -> SimpleNamespace:
    defaults = {
        "active_profile": profile,
        "config": SimpleNamespace(
            request_max_prompt_chars=5000,
            request_max_output_tokens=1024,
            telemetry_mode="noop",
        ),
        "decide_for_client": AsyncMock(),
        "model_registry": SimpleNamespace(
            source="static",
            version="v",
            enabled_models=lambda: [],
            get_by_tier=lambda t: [],
        ),
    }
    defaults.update(methods)
    return SimpleNamespace(**defaults)


@pytest.fixture
def client_factory(monkeypatch):
    monkeypatch.setattr("src.api.rate_limiter", SlidingWindowRateLimiter(limit_per_minute=0))
    app.dependency_overrides[require_router_auth] = lambda: None

    def _factory(service):
        async def _fake_build(**kwargs):
            return service, None if service is not None else "unavailable"

        monkeypatch.setattr("src.api.build_routing_service", _fake_build)
        return TestClient(app)

    yield _factory
    app.dependency_overrides.clear()


def test_messages_requires_passthrough_anthropic_profile(client_factory):
    execute = ClientProfile(
        name="demo",
        mode="execute",
        protocol="openai",
        upstream_base_url="",
        upstream_api_key="",
        models={"cheap": "", "mid": "", "hard": ""},
    )
    with client_factory(_svc(execute)) as client:
        resp = client.post(
            "/v1/messages",
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
        )
    assert resp.status_code == 400
    assert "passthrough" in resp.json()["error"]["message"]


def test_messages_rejects_wrong_protocol(client_factory):
    profile = _anthropic_profile(protocol="openai", name="default")
    with client_factory(_svc(profile)) as client:
        resp = client.post(
            "/v1/messages",
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
        )
    assert resp.status_code == 400
    assert "expected 'anthropic'" in resp.json()["error"]["message"]


def test_messages_forwards_and_injects_dispatch_metadata(client_factory, monkeypatch):
    profile = _anthropic_profile()
    decision = ClientRouteDecision(
        request_id="a-1",
        classification=_classification(Tier.CHEAP),
        tier=Tier.CHEAP,
        selected_model="claude-haiku",
        profile=profile,
        cache_hit=False,
        classification_latency_ms=1.5,
        reason="passthrough",
    )
    svc = _svc(profile, decide_for_client=AsyncMock(return_value=decision))

    async def _forward(**kwargs):
        assert kwargs["selected_model"] == "claude-haiku"
        return PassthroughResult(
            status_code=200,
            headers={"content-type": "application/json"},
            body=b'{"id":"msg_1","content":[{"type":"text","text":"ok"}]}',
            stream=None,
            selected_model="claude-haiku",
            tier=Tier.CHEAP,
            latency_ms=11.0,
            request_id="a-1",
        )

    monkeypatch.setattr("src.anthropic_compat.forward_anthropic_messages", _forward)
    monkeypatch.setattr("src.anthropic_compat.router_api_key_configured", lambda: False)

    from src.api import memory_telemetry

    before = len(memory_telemetry.recent(1000))
    with client_factory(svc) as client:
        resp = client.post(
            "/v1/messages",
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32},
            headers={"x-api-key": "sk-ant-test"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["dispatch"]["tier"] == "cheap"
    assert body["dispatch"]["model"] == "claude-haiku"
    assert resp.headers["X-Dispatch-Tier"] == "cheap"
    assert len(memory_telemetry.recent(1000)) > before


def test_messages_missing_upstream_credentials(client_factory, monkeypatch):
    profile = _anthropic_profile(upstream_api_key="")
    decision = ClientRouteDecision(
        request_id="a-2",
        classification=_classification(),
        tier=Tier.CHEAP,
        selected_model="claude-haiku",
        profile=profile,
        cache_hit=False,
        classification_latency_ms=1.0,
        reason="passthrough",
    )
    svc = _svc(profile, decide_for_client=AsyncMock(return_value=decision))
    monkeypatch.setattr("src.anthropic_compat.router_api_key_configured", lambda: False)
    with client_factory(svc) as client:
        resp = client.post(
            "/v1/messages",
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
        )
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"


def test_messages_upstream_forward_failure(client_factory, monkeypatch):
    profile = _anthropic_profile()
    decision = ClientRouteDecision(
        request_id="a-3",
        classification=_classification(),
        tier=Tier.MID,
        selected_model="claude-sonnet",
        profile=profile,
        cache_hit=False,
        classification_latency_ms=1.0,
        reason="passthrough",
    )
    svc = _svc(profile, decide_for_client=AsyncMock(return_value=decision))

    async def _boom(**kwargs):
        raise ConnectionError("refused")

    monkeypatch.setattr("src.anthropic_compat.forward_anthropic_messages", _boom)
    monkeypatch.setattr("src.anthropic_compat.router_api_key_configured", lambda: False)
    with client_factory(svc) as client:
        resp = client.post(
            "/v1/messages",
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
            headers={"x-api-key": "sk-ant-test"},
        )
    assert resp.status_code == 502
    assert "upstream forward failed" in resp.json()["error"]["message"]


