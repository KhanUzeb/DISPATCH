from __future__ import annotations

import asyncio
import importlib
import sys
from types import ModuleType

import pytest

from src.router.cache import DecisionCacheKey, InMemoryDecisionCache
from src.router.executor import Executor
from src.router.providers.base import ProviderAdapter, ProviderStreamChunk
from src.router.providers.factory import build_provider_registry
from src.router.providers.registry import ProviderRegistry
from src.router.schemas import (
    ClassificationResult,
    ModelSpec,
    Provider,
    ProviderRequest,
    ProviderResponse,
    RoutingCandidate,
    RoutingConstraints,
    RoutingDecision,
    Tier,
)


def _model(key: str) -> ModelSpec:
    return ModelSpec(key, key, Provider.GROQ, Tier.CHEAP, True, 0, 0, 1, 1000, False, False)


def _decision(*models: ModelSpec) -> RoutingDecision:
    return RoutingDecision(
        models[0],
        ClassificationResult("cheap", Tier.CHEAP, 1, True, "c", "r", "e"),
        [RoutingCandidate(m, [], 0, 0) for m in models],
        "test",
        "p",
    )


def test_cache_key_includes_routing_inputs():
    base = dict(namespace="x", prompt="p", constraints={}, route_version="r", encoder_id="e", policy_version="p")
    cache = InMemoryDecisionCache()
    cache.set(DecisionCacheKey(**base), "decision")
    assert cache.get(DecisionCacheKey(**base, expects_structured_output=True)) is None
    constrained = dict(base, constraints={"optimization_objective": "lowest_latency", "allow_providers": ["groq"]})
    cache.set(DecisionCacheKey(**base), "decision")
    assert cache.get(DecisionCacheKey(**constrained)) is None


def test_fallback_limit_counts_only_fallback_models():
    executor = Executor(ProviderRegistry({}), max_fallbacks=0)
    assert [m.key for m in executor._fallback_models(_decision(_model("a"), _model("b")))] == ["a"]
    executor = Executor(ProviderRegistry({}), max_fallbacks=1)
    assert [m.key for m in executor._fallback_models(_decision(_model("a"), _model("b"), _model("c")))] == ["a", "b"]


def test_provider_registry_adapters_share_client():
    client = object()
    registry = build_provider_registry(client=client)  # type: ignore[arg-type]
    assert all(getattr(adapter, "client", client) is client for adapter in registry.providers.values() if hasattr(adapter, "client"))


def test_all_exports_are_defined():
    for module_name in ("src.router", "src.router.models", "src.router.providers"):
        module = importlib.import_module(module_name)
        assert all(hasattr(module, name) for name in module.__all__)


def test_mcp_module_import_has_optional_dependency_guard(monkeypatch):
    import src.mcp_server as server
    assert hasattr(server, "MCPServer")
    # The guarded module must remain importable even when MCP is absent.
    assert server.mcp is None or server.MCPServer is not None


@pytest.mark.asyncio
async def test_stream_usage_is_captured_when_provider_reports_it():
    class StreamingAdapter(ProviderAdapter):
        @property
        def provider_name(self):
            return "groq"

        async def execute(self, model_id, request):
            raise AssertionError

        async def stream_with_usage(self, model_id, request):
            yield ProviderStreamChunk("hello")
            yield ProviderStreamChunk("", usage_input_tokens=7, usage_output_tokens=3)

    result, deltas = await Executor(ProviderRegistry({Provider.GROQ: StreamingAdapter()})).execute_stream(
        _decision(_model("a")), ProviderRequest("hi", [{"role": "user", "content": "hi"}])
    )
    assert deltas is not None
    assert [x async for x in deltas] == ["hello"]
    assert result.stream_usage == {"input_tokens": 7, "output_tokens": 3}
    assert result.response is not None and not result.response.usage_available
