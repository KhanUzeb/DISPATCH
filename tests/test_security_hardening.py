from __future__ import annotations

import threading

import pytest
from fastapi import HTTPException

from src.router.cache import DecisionCacheKey, InMemoryDecisionCache
from src.router.executor import Executor
from src.router.schemas import (
    ClassificationResult,
    ModelSpec,
    Provider,
    ProviderRequest,
    ProviderResponse,
    RoutingCandidate,
    RoutingDecision,
    Tier,
)
from src.router.security import (
    SlidingWindowRateLimiter,
    client_safe_upstream_message,
    is_retryable_provider_error,
    verify_router_api_key,
)


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
        provider_model_id="llama",
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


def _decision(model: ModelSpec | None = None) -> RoutingDecision:
    selected = model
    candidates = []
    if selected is not None:
        candidates = [
            RoutingCandidate(
                model=selected,
                rejected_reasons=[],
                estimated_cost_usd=0.0,
                objective_score=0.0,
            )
        ]
    return RoutingDecision(
        selected_model=selected,
        classification=_classification(),
        candidates=candidates,
        reason="test",
        policy_version="v2",
    )


def test_verify_api_key_optional(monkeypatch):
    monkeypatch.delenv("ROUTER_API_KEY", raising=False)
    verify_router_api_key(None)
    verify_router_api_key("Bearer anything")


def test_verify_api_key_required(monkeypatch):
    monkeypatch.setenv("ROUTER_API_KEY", "secret-demo")
    with pytest.raises(HTTPException) as missing:
        verify_router_api_key(None)
    assert missing.value.status_code == 401

    with pytest.raises(HTTPException) as bad:
        verify_router_api_key("Bearer wrong")
    assert bad.value.status_code == 401

    verify_router_api_key("Bearer secret-demo")


def test_rate_limiter_blocks_after_limit():
    limiter = SlidingWindowRateLimiter(limit_per_minute=3)
    assert limiter.allow("a")
    assert limiter.allow("a")
    assert limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.allow("b")


def test_rate_limiter_disabled():
    limiter = SlidingWindowRateLimiter(limit_per_minute=0)
    for _ in range(20):
        assert limiter.allow("a")


def test_decision_cache_evicts_and_is_thread_safe():
    cache = InMemoryDecisionCache(ttl_seconds=60, max_entries=5)

    def _key(i: int) -> DecisionCacheKey:
        return DecisionCacheKey(
            namespace="default",
            prompt=f"prompt-{i}",
            constraints={},
            route_version="r1",
            encoder_id="fake",
            policy_version="v2",
        )

    decision = _decision()
    errors: list[BaseException] = []

    def writer(start: int, count: int) -> None:
        try:
            for i in range(start, start + count):
                cache.set(_key(i), decision)
                cache.get(_key(i))
        except BaseException as exc:  # noqa: BLE001 — collect for assertion
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i * 10, 10)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(cache._store) <= 5


def test_executor_retries_then_succeeds():
    class Flaky:
        provider_name = "groq"

        def __init__(self) -> None:
            self.calls = 0

        def execute(self, model_id: str, request: ProviderRequest) -> ProviderResponse:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("rate limited")
            return ProviderResponse(
                text="ok",
                usage_input_tokens=1,
                usage_output_tokens=1,
                latency_ms=1.0,
                model=model_id,
                provider=Provider.GROQ,
            )

    class Registry:
        def __init__(self, adapter):
            self.adapter = adapter

        def get(self, provider: Provider):
            return self.adapter

    model = _model()
    adapter = Flaky()
    executor = Executor(Registry(adapter), max_fallbacks=0, retries_by_provider={"groq": 1})
    result = executor.execute(
        _decision(model),
        ProviderRequest(prompt="hi", messages=[{"role": "user", "content": "hi"}], max_output_tokens=16),
    )
    assert result.response is not None
    assert result.response.text == "ok"
    assert adapter.calls == 2


def test_executor_hides_raw_upstream_errors():
    class Boom:
        provider_name = "groq"

        def execute(self, model_id: str, request: ProviderRequest) -> ProviderResponse:
            raise RuntimeError("secret stacktrace with api key sk-abc")

    class Registry:
        def get(self, provider: Provider):
            return Boom()

    result = Executor(Registry(), max_fallbacks=0).execute(
        _decision(_model()),
        ProviderRequest(prompt="hi", messages=[{"role": "user", "content": "hi"}], max_output_tokens=16),
    )
    assert result.error is not None
    assert "sk-abc" not in result.error.message
    assert result.error.message == client_safe_upstream_message("m1")


def test_retryable_detection():
    assert is_retryable_provider_error(RuntimeError("429 rate limited"))
    assert not is_retryable_provider_error(RuntimeError("invalid api key"))
