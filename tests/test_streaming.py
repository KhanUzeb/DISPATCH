from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from src.openai_compat import iter_sse_chunks
from src.router.executor import Executor
from src.router.providers.openai_compatible import (
    OpenAICompatibleAdapter,
    _delta_text_from_chunk,
    _parse_sse_data_line,
)
from src.router.providers.registry import ProviderRegistry
from src.router.schemas import (
    ClassificationResult,
    ModelSpec,
    Provider,
    ProviderRequest,
    RoutingCandidate,
    RoutingDecision,
    Tier,
)


def _model(key: str = "m1", provider: Provider = Provider.GROQ) -> ModelSpec:
    return ModelSpec(
        key=key,
        provider_model_id=f"{key}-id",
        provider=provider,
        tier=Tier.CHEAP,
        enabled=True,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        avg_latency_ms=50,
        context_window=8000,
        supports_tool_calling=False,
        supports_structured_output=False,
    )


def _decision(*models: ModelSpec) -> RoutingDecision:
    selected = models[0]
    return RoutingDecision(
        selected_model=selected,
        classification=ClassificationResult(
            route="cheap",
            tier=Tier.CHEAP,
            similarity_score=0.9,
            passed_threshold=True,
            classifier_version="c1",
            route_config_version="r1",
            encoder_id="fake:64",
        ),
        candidates=[
            RoutingCandidate(
                model=m,
                rejected_reasons=[],
                estimated_cost_usd=0.0,
                objective_score=0.0,
            )
            for m in models
        ],
        reason="test",
        policy_version="v2",
    )


@pytest.mark.asyncio
async def test_execute_stream_yields_incremental_deltas():
    class StreamingAdapter:
        provider_name = "groq"

        async def execute(self, model_id, request):
            raise AssertionError("execute should not be called")

        async def stream(self, model_id, request) -> AsyncIterator[str]:
            for part in ("a", "b", "c"):
                yield part
                await asyncio.sleep(0)

    executor = Executor(ProviderRegistry(providers={Provider.GROQ: StreamingAdapter()}))
    result, deltas = await executor.execute_stream(
        _decision(_model()),
        ProviderRequest(prompt="hi", messages=[{"role": "user", "content": "hi"}]),
    )
    assert result.error is None
    assert result.used_model is not None
    assert deltas is not None
    assert [p async for p in deltas] == ["a", "b", "c"]


