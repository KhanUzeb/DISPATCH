"""FastAPI route handlers via TestClient (no real network)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.router.clients import ClientProfile
from src.router.executor import ExecutionResult
from src.router.schemas import (
    ClassificationResult,
    ModelSpec,
    Provider,
    ProviderResponse,
    RoutingCandidate,
    RoutingDecision,
    RoutingError,
    RoutingErrorCategory,
    Tier,
)
from src.router.security import SlidingWindowRateLimiter, require_router_auth
from src.router.service import ClientRouteDecision, RouteResponse


def _classification(tier: Tier = Tier.CHEAP) -> ClassificationResult:
    return ClassificationResult(
        route=tier.value,
        tier=tier,
        similarity_score=0.91,
        passed_threshold=True,
        classifier_version="c1",
        route_config_version="r1",
        encoder_id="fake:64",
    )


def _model(tier: Tier = Tier.CHEAP) -> ModelSpec:
    return ModelSpec(
        key="m1",
        provider_model_id="llama-test",
        provider=Provider.GROQ,
        tier=tier,
        enabled=True,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        avg_latency_ms=100,
        context_window=8000,
        supports_tool_calling=False,
        supports_structured_output=False,
    )


def _route_response(model: ModelSpec | None = None) -> RouteResponse:
    selected = model or _model()
    decision = RoutingDecision(
        selected_model=selected,
        classification=_classification(selected.tier),
        candidates=[
            RoutingCandidate(
                model=selected,
                rejected_reasons=[],
                estimated_cost_usd=0.01,
                objective_score=1.0,
            )
        ],
        reason="test-route",
        policy_version="v2",
    )
    return RouteResponse(
        request_id="req-1",
        decision=decision,
        cache_hit=False,
        classification_latency_ms=1.0,
        policy_latency_ms=0.5,
    )


def _execute_profile() -> ClientProfile:
    return ClientProfile(
        name="demo",
        mode="execute",
        protocol="openai",
        upstream_base_url="",
        upstream_api_key="",
        models={"cheap": "", "mid": "", "hard": ""},
    )


def _passthrough_profile(*, protocol: str = "openai") -> ClientProfile:
    return ClientProfile(
        name="default" if protocol == "openai" else "anthropic",
        mode="passthrough",
        protocol=protocol,
        upstream_base_url="https://example.com/v1",
        upstream_api_key="sk-test",
        models={"cheap": "c", "mid": "m", "hard": "h"},
        discover_models=False,
    )


class _FakeRegistry:
    source = "static"
    version = "v-test"

    def enabled_models(self):
        return [_model()]

    def get_by_tier(self, tier: Tier):
        return [_model(tier)]

    async def refresh(self, *, client):
        return self


def _make_service(*, profile: ClientProfile | None = None, **methods) -> SimpleNamespace:
    defaults = {
        "active_profile": profile,
        "config": SimpleNamespace(
            request_max_prompt_chars=1000,
            request_max_output_tokens=4096,
            telemetry_mode="noop",
        ),
        "model_registry": _FakeRegistry(),
        "decide": lambda *a, **k: _route_response(),
        "decide_for_client": AsyncMock(),
        "complete": AsyncMock(),
    }
    defaults.update(methods)
    return SimpleNamespace(**defaults)


@pytest.fixture
def client_factory(monkeypatch):
    """Build a TestClient whose lifespan returns a controlled service."""

    monkeypatch.setattr("src.api.rate_limiter", SlidingWindowRateLimiter(limit_per_minute=0))
    app.dependency_overrides[require_router_auth] = lambda: None

    def _factory(service, startup_error: str | None = None):
        async def _fake_build(**kwargs):
            return service, startup_error

        monkeypatch.setattr("src.api.build_routing_service", _fake_build)
        return TestClient(app)

    yield _factory
    app.dependency_overrides.clear()


def test_ready_true(client_factory, monkeypatch):
    profile = _execute_profile()
    profile.ensure_models = AsyncMock()  # type: ignore[method-assign]
    svc = _make_service(profile=profile)
    monkeypatch.setattr(
        "src.api.ready_payload",
        AsyncMock(return_value={"ready": True, "profile": "demo", "models": []}),
    )
    with client_factory(svc) as client:
        resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True


def test_models_refresh(client_factory):
    svc = _make_service(profile=_execute_profile())
    with client_factory(svc) as client:
        resp = client.post("/models/refresh")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["source"] == "static"
    assert "cheap" in body["tiers"]


def test_route_execute_mode(client_factory):
    svc = _make_service(profile=_execute_profile())
    with client_factory(svc) as client:
        resp = client.post("/route", json={"prompt": "summarize this", "include_diagnostics": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "execute"
    assert body["provider_model_id"] == "llama-test"
    assert "classification" in body
    assert "candidates" in body


def test_route_passthrough_mode(client_factory):
    profile = _passthrough_profile()
    decision = ClientRouteDecision(
        request_id="pt-1",
        classification=_classification(Tier.MID),
        tier=Tier.MID,
        selected_model="m",
        profile=profile,
        cache_hit=False,
        classification_latency_ms=2.0,
        reason="passthrough:default classified=mid tier=mid → m",
    )
    svc = _make_service(
        profile=profile,
        decide_for_client=AsyncMock(return_value=decision),
    )
    with client_factory(svc) as client:
        resp = client.post(
            "/route",
            json={"prompt": "explain this function", "include_diagnostics": True},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "passthrough"
    assert body["profile"] == "default"
    assert body["provider_model_id"] == "m"
    assert body["tier_models"]["mid"] == "m"
    assert "classification" in body


def test_route_rejects_oversized_prompt(client_factory):
    svc = _make_service(profile=_execute_profile())
    svc.config.request_max_prompt_chars = 5
    with client_factory(svc) as client:
        resp = client.post("/route", json={"prompt": "too long prompt"})
    assert resp.status_code == 400
    assert resp.json()["detail"]["message"] == "prompt exceeds limit"


def test_complete_success(client_factory):
    model = _model()
    route_result = _route_response(model)
    execution = ExecutionResult(
        response=ProviderResponse(
            text="hello",
            usage_input_tokens=3,
            usage_output_tokens=1,
            latency_ms=12.0,
            model=model.provider_model_id,
            provider=Provider.GROQ,
        ),
        decision=route_result.decision,
        fallback_attempts=[],
        actual_cost_usd=0.0,
        used_model=model,
    )
    svc = _make_service(
        profile=_execute_profile(),
        complete=AsyncMock(return_value=(route_result, execution)),
    )
    with client_factory(svc) as client:
        resp = client.post("/complete", json={"prompt": "hi there", "max_output_tokens": 32})
    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "hello"
    assert body["provider"] == "groq"
    assert body["usage"]["input_tokens"] == 3


def test_complete_rejects_passthrough(client_factory):
    svc = _make_service(profile=_passthrough_profile(protocol="anthropic"))
    with client_factory(svc) as client:
        resp = client.post("/complete", json={"prompt": "hello"})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["error_category"] == "invalid_request"
    assert "/v1/messages" in detail["message"]


def test_complete_provider_error(client_factory):
    route_result = _route_response()
    execution = ExecutionResult(
        response=None,
        decision=route_result.decision,
        fallback_attempts=["m1"],
        error=RoutingError(
            category=RoutingErrorCategory.PROVIDER_UNAVAILABLE,
            message="upstream down",
            retryable=True,
        ),
    )
    svc = _make_service(
        profile=_execute_profile(),
        complete=AsyncMock(return_value=(route_result, execution)),
    )
    with client_factory(svc) as client:
        resp = client.post("/complete", json={"prompt": "hello"})
    assert resp.status_code == 502
    assert resp.json()["detail"]["error_category"] == "provider_unavailable"


def test_dashboard_and_telemetry_recent(client_factory, monkeypatch):
    from src.api import memory_telemetry
    from src.router.telemetry import RoutingTelemetryEvent

    svc = _make_service(profile=_execute_profile())
    with client_factory(svc) as client:
        dash = client.get("/dashboard")
        assert dash.status_code == 200
        assert "text/html" in dash.headers.get("content-type", "")
        assert b"Routing" in dash.content or b"Dashboard" in dash.content

        memory_telemetry.emit(
            RoutingTelemetryEvent(
                request_id="dash-1",
                route="cheap",
                selected_model="llama-test",
                selected_provider="groq",
                tier="cheap",
                cache_hit=False,
                classification_latency_ms=1.0,
                policy_latency_ms=0.5,
                provider_latency_ms=10.0,
                total_latency_ms=12.0,
                estimated_cost_usd=None,
                actual_cost_usd=None,
                input_tokens=4,
                output_tokens=8,
                fallback_attempts=0,
                error_category=None,
                route_confidence=0.9,
                config_versions={},
            )
        )
        recent = client.get("/telemetry/recent?limit=10")
    assert recent.status_code == 200
    body = recent.json()
    assert body["count"] >= 1
    assert any(e.get("request_id") == "dash-1" for e in body["events"])

