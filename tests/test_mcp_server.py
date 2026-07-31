"""MCP tool surface: route / complete / refresh_models / health / ready."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.mcp_server import (
    complete,
    health,
    ready,
    refresh_models,
    route,
)
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
from src.router.service import ClientRouteDecision, RouteResponse
from src.router.surface import RequestLimitError


def _classification(tier: Tier = Tier.CHEAP) -> ClassificationResult:
    return ClassificationResult(
        route=tier.value,
        tier=tier,
        similarity_score=0.9,
        passed_threshold=True,
        classifier_version="c1",
        route_config_version="r1",
        encoder_id="fake:64",
    )


def _model() -> ModelSpec:
    return ModelSpec(
        key="m1",
        provider_model_id="llama-test",
        provider=Provider.GROQ,
        tier=Tier.CHEAP,
        enabled=True,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        avg_latency_ms=100,
        context_window=8000,
        supports_tool_calling=False,
        supports_structured_output=False,
    )


def _route_response() -> RouteResponse:
    model = _model()
    decision = RoutingDecision(
        selected_model=model,
        classification=_classification(),
        candidates=[
            RoutingCandidate(
                model=model,
                rejected_reasons=[],
                estimated_cost_usd=0.0,
                objective_score=1.0,
            )
        ],
        reason="mcp-test",
        policy_version="v2",
    )
    return RouteResponse(
        request_id="mcp-1",
        decision=decision,
        cache_hit=False,
        classification_latency_ms=1.0,
        policy_latency_ms=0.5,
    )


def _execute_svc(**methods) -> SimpleNamespace:
    defaults = {
        "active_profile": ClientProfile(
            name="demo",
            mode="execute",
            protocol="openai",
            upstream_base_url="",
            upstream_api_key="",
            models={"cheap": "", "mid": "", "hard": ""},
        ),
        "config": SimpleNamespace(request_max_prompt_chars=1000, request_max_output_tokens=4096),
        "decide": lambda *a, **k: _route_response(),
        "complete": AsyncMock(),
        "model_registry": SimpleNamespace(
            source="static",
            version="v1",
            enabled_models=lambda: [_model()],
            get_by_tier=lambda tier: [_model()],
            refresh=AsyncMock(
                return_value=SimpleNamespace(
                    source="static",
                    version="v2",
                    enabled_models=lambda: [_model()],
                    get_by_tier=lambda tier: [_model()],
                )
            ),
        ),
    }
    defaults.update(methods)
    return SimpleNamespace(**defaults)


@pytest.fixture
def patch_service(monkeypatch):
    def _apply(svc=None, *, error: str | None = None):
        if error is not None:
            monkeypatch.setattr(
                "src.mcp_server._service_or_error",
                lambda: (_ for _ in ()).throw(RuntimeError(error)),
            )
            monkeypatch.setattr("src.mcp_server._ensure_service", lambda: None)
            monkeypatch.setattr("src.mcp_server._service", None)
            monkeypatch.setattr("src.mcp_server._startup_error", error)
            return

        monkeypatch.setattr("src.mcp_server._service_or_error", lambda: svc)
        monkeypatch.setattr("src.mcp_server._ensure_service", lambda: None)
        monkeypatch.setattr("src.mcp_server._service", svc)
        monkeypatch.setattr("src.mcp_server._startup_error", None)

    return _apply


def _stub_run(result):
    """Replace _run so async tool paths stay sync without leaving dangling coros."""

    def _run(coro):
        if hasattr(coro, "close"):
            coro.close()
        return result

    return _run


def test_route_execute(patch_service):
    patch_service(_execute_svc())
    payload = route(prompt="summarize this", include_diagnostics=True)
    assert payload["mode"] == "execute"
    assert payload["provider_model_id"] == "llama-test"
    assert "classification" in payload
    assert "candidates" in payload


def test_route_passthrough(patch_service, monkeypatch):
    profile = ClientProfile(
        name="anthropic",
        mode="passthrough",
        protocol="anthropic",
        upstream_base_url="https://api.anthropic.com",
        upstream_api_key="sk",
        models={"cheap": "a", "mid": "b", "hard": "c"},
        discover_models=False,
    )
    decision = ClientRouteDecision(
        request_id="pt",
        classification=_classification(Tier.HARD),
        tier=Tier.HARD,
        selected_model="c",
        profile=profile,
        cache_hit=False,
        classification_latency_ms=2.0,
        reason="passthrough",
    )
    svc = _execute_svc(active_profile=profile, decide_for_client=AsyncMock(return_value=decision))
    patch_service(svc)
    monkeypatch.setattr("src.mcp_server._run", _stub_run(decision))
    monkeypatch.setattr("src.mcp_server.get_http_client", lambda: AsyncMock())

    payload = route(prompt="design a system", include_diagnostics=True)
    assert payload["mode"] == "passthrough"
    assert payload["provider_model_id"] == "c"
    assert payload["tier"] == "hard"
    assert "classification" in payload


def test_complete_success(patch_service, monkeypatch):
    route_result = _route_response()
    model = route_result.decision.selected_model
    execution = ExecutionResult(
        response=ProviderResponse(
            text="done",
            usage_input_tokens=2,
            usage_output_tokens=1,
            latency_ms=9.0,
            model=model.provider_model_id,
            provider=Provider.GROQ,
        ),
        decision=route_result.decision,
        used_model=model,
        actual_cost_usd=0.0,
    )
    svc = _execute_svc(complete=AsyncMock(return_value=(route_result, execution)))
    patch_service(svc)
    monkeypatch.setattr("src.mcp_server._run", _stub_run((route_result, execution)))

    payload = complete(prompt="hello", max_output_tokens=32)
    assert payload["response"] == "done"
    assert payload["provider"] == "groq"


def test_complete_provider_error(patch_service, monkeypatch):
    route_result = _route_response()
    execution = ExecutionResult(
        response=None,
        decision=route_result.decision,
        fallback_attempts=["m1"],
        error=RoutingError(
            category=RoutingErrorCategory.PROVIDER_TIMEOUT,
            message="timed out",
            retryable=True,
        ),
    )
    patch_service(_execute_svc())
    monkeypatch.setattr("src.mcp_server._run", _stub_run((route_result, execution)))
    payload = complete(prompt="hello")
    assert payload["error"]["category"] == "provider_timeout"
    assert payload["fallback_attempts"] == ["m1"]


def test_complete_passthrough_rejected(patch_service):
    profile = ClientProfile(
        name="default",
        mode="passthrough",
        protocol="openai",
        upstream_base_url="https://openrouter.ai/api/v1",
        upstream_api_key="",
        models={"cheap": "a", "mid": "b", "hard": "c"},
    )
    patch_service(_execute_svc(active_profile=profile))
    payload = complete(prompt="hello")
    assert payload["error"]["category"] == "invalid_request"
    assert "/v1/chat/completions" in payload["error"]["message"]


def test_refresh_models(patch_service, monkeypatch):
    refreshed = SimpleNamespace(
        source="discover",
        version="v-new",
        enabled_models=lambda: [_model()],
        get_by_tier=lambda tier: [_model()],
    )
    svc = _execute_svc()
    patch_service(svc)
    monkeypatch.setattr("src.mcp_server._run", _stub_run(refreshed))
    monkeypatch.setattr("src.mcp_server.get_http_client", lambda: AsyncMock())

    payload = refresh_models()
    assert payload["ok"] is True
    assert payload["source"] == "discover"
    assert payload["version"] == "v-new"
    assert svc.model_registry is refreshed


