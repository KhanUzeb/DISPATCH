from pathlib import Path

from src.router.schemas import ClassificationResult, RoutingDecision, Tier
from src.router.telemetry import InMemoryTelemetry, LangfuseTelemetry, RoutingTelemetryEvent


def _event(request_id: str = "req-1") -> RoutingTelemetryEvent:
    classification = ClassificationResult(
        route="cheap",
        tier=Tier.CHEAP,
        similarity_score=0.9,
        passed_threshold=True,
        classifier_version="v2",
        route_config_version="r1",
        encoder_id="fake:64",
    )
    decision = RoutingDecision(
        selected_model=None,
        classification=classification,
        candidates=[],
        reason="test",
        policy_version="v2",
    )
    return RoutingTelemetryEvent(
        request_id=request_id,
        route="cheap",
        selected_model="llama-3.1-8b-instant",
        selected_provider="groq",
        tier="cheap",
        cache_hit=False,
        classification_latency_ms=1.0,
        policy_latency_ms=1.0,
        provider_latency_ms=10.0,
        total_latency_ms=12.0,
        estimated_cost_usd=0.0,
        actual_cost_usd=0.0,
        input_tokens=5,
        output_tokens=3,
        fallback_attempts=0,
        error_category=None,
        route_confidence=0.9,
        config_versions={"policy_version": "v2"},
        decision=decision,
    )


def test_langfuse_emit_uses_start_observation():
    calls = {}

    class FakeObservation:
        def end(self):
            calls["ended"] = True

    class FakeLangfuse:
        def start_observation(self, **kwargs):
            calls["start"] = kwargs
            return FakeObservation()

        def flush(self):
            calls["flushed"] = True

    sink = LangfuseTelemetry()
    sink._get_client = lambda: FakeLangfuse()  # type: ignore[method-assign]
    sink.emit(_event())

    assert calls["start"]["name"] == "dispatch-routing"
    assert calls["start"]["as_type"] == "generation"
    assert calls["ended"] is True
    assert calls["flushed"] is True


def test_inmemory_telemetry_persists_across_instances(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"
    first = InMemoryTelemetry(max_events=100, persist_path=path)
    first.emit(_event("a"))
    first.emit(_event("b"))

    second = InMemoryTelemetry(max_events=100, persist_path=path)
    recent = second.recent(limit=10)
    assert len(recent) == 2
    assert {e["request_id"] for e in recent} == {"a", "b"}
    assert all("decision" not in e for e in recent)
