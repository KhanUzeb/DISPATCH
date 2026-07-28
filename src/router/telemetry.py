from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .schemas import RoutingDecision, to_dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingTelemetryEvent:
    request_id: str
    route: str | None
    selected_model: str | None
    selected_provider: str | None
    tier: str
    cache_hit: bool
    classification_latency_ms: float
    policy_latency_ms: float
    provider_latency_ms: float
    total_latency_ms: float
    estimated_cost_usd: float | None
    actual_cost_usd: float | None
    input_tokens: int
    output_tokens: int
    fallback_attempts: int
    error_category: str | None
    route_confidence: float
    config_versions: dict[str, str]
    decision: RoutingDecision | None = None
    created_at_ms: int = 0

    def __post_init__(self) -> None:
        if not self.created_at_ms:
            object.__setattr__(self, "created_at_ms", int(time.time() * 1000))


class TelemetrySink(Protocol):
    def emit(self, event: RoutingTelemetryEvent) -> None:
        ...


class NoopTelemetry:
    def emit(self, event: RoutingTelemetryEvent) -> None:
        return


class LoggingTelemetry:
    REDACTED_KEYS = {"prompt", "api_key", "authorization", "token", "secret"}

    def emit(self, event: RoutingTelemetryEvent) -> None:
        payload = to_dict(event)
        if payload.get("decision") is not None:
            payload["decision"] = self._redact(payload["decision"])
        logger.info("routing_decision_event=%s", payload)

    def _redact(self, obj):
        if isinstance(obj, dict):
            redacted = {}
            for k, v in obj.items():
                if k.lower() in self.REDACTED_KEYS:
                    redacted[k] = "***redacted***"
                else:
                    redacted[k] = self._redact(v)
            return redacted
        if isinstance(obj, list):
            return [self._redact(x) for x in obj]
        return obj


class InMemoryTelemetry:
    """In-process ring buffer with optional JSONL persistence on disk.

    Persistence survives uvicorn --reload and API restarts so the dashboard
    keeps showing recent routing events.
    """

    def __init__(self, max_events: int = 500, persist_path: str | Path | None = None) -> None:
        self._events: deque[dict] = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path is not None:
            self._load_persisted()

    def _load_persisted(self) -> None:
        assert self._persist_path is not None
        if not self._persist_path.exists():
            return
        try:
            loaded: list[dict] = []
            with self._persist_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        loaded.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            for item in loaded[-self._events.maxlen :]:  # type: ignore[index]
                self._events.append(item)
        except Exception:
            logger.exception("Failed to load persisted telemetry from %s", self._persist_path)

    def _append_persist(self, payload: dict) -> None:
        if self._persist_path is None:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with self._persist_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")
            # Compact file if it grows too large.
            if self._persist_path.stat().st_size > 2_000_000:
                self._compact_persist()
        except Exception:
            logger.exception("Failed to persist telemetry event")

    def _compact_persist(self) -> None:
        assert self._persist_path is not None
        tmp = self._persist_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for item in list(self._events):
                fh.write(json.dumps(item, default=str) + "\n")
        tmp.replace(self._persist_path)

    def emit(self, event: RoutingTelemetryEvent) -> None:
        payload = to_dict(event)
        # Keep dashboard payloads light; full decision stays in other sinks.
        payload.pop("decision", None)
        with self._lock:
            self._events.append(payload)
            self._append_persist(payload)

    def recent(self, limit: int = 100) -> list[dict]:
        with self._lock:
            return list(self._events)[-limit:]


class LangfuseTelemetry:
    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        if not public_key or not secret_key:
            raise RuntimeError("Langfuse keys are missing")
        from langfuse import Langfuse

        host = (
            os.environ.get("LANGFUSE_HOST")
            or os.environ.get("LANGFUSE_BASE_URL")
            or "https://cloud.langfuse.com"
        )
        self._client = Langfuse(
            public_key=public_key.strip('"'),
            secret_key=secret_key.strip('"'),
            host=host.strip('"'),
        )
        return self._client

    def emit(self, event: RoutingTelemetryEvent) -> None:
        try:
            client = self._get_client()
            payload = to_dict(event)
            payload.pop("decision", None)

            # Langfuse SDK v3+/v4 no longer has client.trace(); use observations.
            observation = client.start_observation(
                name="dispatch-routing",
                as_type="generation",
                metadata=payload,
                model=event.selected_model,
                usage_details={
                    "input": int(event.input_tokens or 0),
                    "output": int(event.output_tokens or 0),
                    "total": int((event.input_tokens or 0) + (event.output_tokens or 0)),
                },
                cost_details=(
                    {"total": float(event.actual_cost_usd)}
                    if event.actual_cost_usd is not None
                    else None
                ),
                level="ERROR" if event.error_category else "DEFAULT",
                status_message=event.error_category,
                output={
                    "tier": event.tier,
                    "provider": event.selected_provider,
                    "model": event.selected_model,
                    "cache_hit": event.cache_hit,
                    "request_id": event.request_id,
                },
                input={
                    "route": event.route,
                    "route_confidence": event.route_confidence,
                    "config_versions": event.config_versions,
                },
            )
            observation.end()
            # Best-effort flush; never block user request on observability.
            try:
                client.flush()
            except Exception:
                pass
        except Exception:
            logger.exception("Failed to emit telemetry to Langfuse")


class MultiTelemetry:
    def __init__(self, sinks: list[TelemetrySink]) -> None:
        self.sinks = sinks

    def emit(self, event: RoutingTelemetryEvent) -> None:
        for sink in self.sinks:
            try:
                sink.emit(event)
            except Exception:
                logger.exception("Telemetry sink failed")
