"""Dispatch FastAPI app — chat demo, dashboard, /route, /complete, and /v1 proxies.

Surfaces:
  GET  / /demo              chat UI (src/chat.html)
  GET  /dashboard           routing & generation telemetry UI
  POST /route /complete     native routing + execution
  GET  /telemetry/recent    recent routing events for the dashboard
  /v1/*                     OpenAI + Anthropic-compatible HTTP
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from .anthropic_compat import register_anthropic_routes
from .openai_compat import register_openai_routes
from .router.bootstrap import build_routing_service
from .router.config import max_request_bytes
from .router.http_client import aclose_http_client, create_http_client, get_http_client, set_http_client
from .router.logging_config import clear_request_id, setup_logging
from .router.schemas import RoutingConstraints, Tier
from .router.security import SlidingWindowRateLimiter, require_router_auth
from .router.service import RoutingService
from .router.surface import RequestLimitError, ready_payload, validate_request_limits
from .router.telemetry import (
    InMemoryTelemetry,
    LangfuseTelemetry,
    LoggingTelemetry,
    MultiTelemetry,
    NoopTelemetry,
    RoutingTelemetryEvent,
)

logger = logging.getLogger(__name__)


class CompleteRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_cost_usd: float | None = Field(default=None, gt=0)
    max_latency_ms: float | None = Field(default=None, gt=0)
    min_context_window: int | None = Field(default=None, gt=0)
    max_output_tokens: int = Field(default=32768, gt=0)
    expects_structured_output: bool = False
    require_tool_calling: bool = False
    include_diagnostics: bool = False

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must not be blank")
        return v.strip()


class ErrorResponse(BaseModel):
    error_category: str
    message: str


service: RoutingService | None = None
startup_error: str | None = None
telemetry = NoopTelemetry()
memory_telemetry = InMemoryTelemetry(
    max_events=1000,
    persist_path=Path(__file__).resolve().parents[1] / ".dispatch_metrics.jsonl",
)
rate_limiter = SlidingWindowRateLimiter.from_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service, startup_error, telemetry
    setup_logging()
    http_client = create_http_client()
    set_http_client(http_client)
    app.state.http_client = http_client
    try:
        service, startup_error = await build_routing_service(
            http_client=http_client,
            log_startup=True,
        )
        if service is None:
            logger.error("%s", startup_error)
        else:
            active = service.active_profile
            registry = service.model_registry
            tier_counts = {t.value: len(registry.get_by_tier(t)) for t in Tier}
            profile_bits = ""
            if active is not None:
                profile_bits = (
                    f"profile={active.name} mode={active.mode} protocol={active.protocol} "
                    + (f"upstream={active.upstream_base_url} " if active.is_passthrough else "")
                )
            logger.info(
                "%smodels_source=%s tiers=%s",
                profile_bits,
                registry.source,
                tier_counts,
            )
            for spec in registry.enabled_models()[:12]:
                logger.info(
                    "  %-5s %-12s %s",
                    spec.tier.value,
                    spec.provider.value,
                    spec.provider_model_id,
                )
            if len(registry.enabled_models()) > 12:
                logger.info("  … +%s more", len(registry.enabled_models()) - 12)

            mode = (service.config.telemetry_mode or "noop").lower()
            sinks: list = [memory_telemetry]
            if mode == "logging":
                sinks.append(LoggingTelemetry())
            elif mode == "langfuse":
                sinks.append(LangfuseTelemetry())
            elif mode == "langfuse+logging":
                sinks.append(LoggingTelemetry())
                sinks.append(LangfuseTelemetry())
            telemetry = MultiTelemetry(sinks=sinks)
        yield
    finally:
        await aclose_http_client()
        app.state.http_client = None


app = FastAPI(
    title="Dispatch",
    description="Semantic LLM router — chat demo, dashboard, MCP, and OpenAI/Anthropic-compatible HTTP.",
    lifespan=lifespan,
)


def _service_or_raise() -> RoutingService:
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error_category="service_unavailable",
                message="service unavailable",
            ).model_dump(),
        )
    return service


def _validate_request_limits(svc: RoutingService, *, prompt: str, max_output_tokens: int) -> None:
    try:
        validate_request_limits(svc, prompt=prompt, max_output_tokens=max_output_tokens)
    except RequestLimitError as exc:
        message = str(exc)
        if "prompt exceeds" in message:
            message = "prompt exceeds limit"
        elif "max_output_tokens exceeds" in message:
            message = "max_output_tokens exceeds limit"
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(error_category="invalid_request", message=message).model_dump(),
        ) from exc


def _emit_telemetry(event: RoutingTelemetryEvent) -> None:
    try:
        telemetry.emit(event)
    except Exception:
        return


@app.middleware("http")
async def clear_request_id_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    finally:
        clear_request_id()


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    if path in {"/health", "/ready", "/demo", "/", "/dashboard"}:
        return await call_next(request)
    client = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(client):
        return JSONResponse(
            status_code=429,
            content={
                "error_category": "rate_limited",
                "message": "Too many requests. Set ROUTER_RATE_LIMIT_PER_MINUTE=0 to disable locally.",
            },
        )
    return await call_next(request)


@app.middleware("http")
async def max_request_size_middleware(request: Request, call_next):
    """Reject oversized bodies via Content-Length before Pydantic parsing."""
    if request.method in {"POST", "PUT", "PATCH"}:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                length = int(content_length)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error_category": "invalid_request",
                        "message": "Invalid Content-Length header",
                    },
                )
            limit = max_request_bytes()
            if length > limit:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error_category": "payload_too_large",
                        "message": "Request body exceeds configured maximum size",
                    },
                )
    return await call_next(request)


app.include_router(register_openai_routes(get_service=_service_or_raise, emit_telemetry=_emit_telemetry))
app.include_router(register_anthropic_routes(get_service=_service_or_raise, emit_telemetry=_emit_telemetry))


@app.get("/")
@app.get("/demo")
def chat_demo():
    html_path = Path(__file__).resolve().parent / "chat.html"
    return FileResponse(str(html_path), media_type="text/html")


@app.get("/dashboard")
def dashboard():
    html_path = Path(__file__).resolve().parent / "dashboard.html"
    return FileResponse(str(html_path), media_type="text/html")


@app.get("/telemetry/recent")
def telemetry_recent(limit: int = 100, _: None = Depends(require_router_auth)):
    """Recent routing/generation events for the dashboard UI."""
    capped = max(1, min(int(limit or 100), 500))
    events = memory_telemetry.recent(limit=capped)
    return {"count": len(events), "events": events}


@app.get("/health")
def health():
    profile = service.active_profile if service else None
    return {
        "status": "ok",
        "profile": profile.name if profile else None,
        "mode": profile.mode if profile else None,
    }


@app.get("/ready")
async def ready():
    if service is None:
        # Avoid leaking absolute filesystem paths to clients.
        return {"ready": False, "error": "startup failed — check server logs"}
    return await ready_payload(service, client=get_http_client())


@app.post("/models/refresh")
async def refresh_models(_: None = Depends(require_router_auth)):
    """Re-fetch live catalogs from Groq / OpenRouter / configured sources."""
    svc = _service_or_raise()
    refreshed = await svc.model_registry.refresh(client=get_http_client())
    svc.model_registry = refreshed
    return {
        "ok": True,
        "source": refreshed.source,
        "version": refreshed.version,
        "count": len(refreshed.enabled_models()),
        "tiers": {
            "cheap": [m.provider_model_id for m in refreshed.get_by_tier(Tier.CHEAP)],
            "mid": [m.provider_model_id for m in refreshed.get_by_tier(Tier.MID)],
            "hard": [m.provider_model_id for m in refreshed.get_by_tier(Tier.HARD)],
        },
    }


@app.post("/route")
async def route_only(req: CompleteRequest, _: None = Depends(require_router_auth)):
    svc = _service_or_raise()
    _validate_request_limits(svc, prompt=req.prompt, max_output_tokens=req.max_output_tokens)
    profile = svc.active_profile
    started = time.perf_counter()

    # Passthrough profiles: return the client's mapped model (no Dispatch provider pick).
    if profile is not None and profile.is_passthrough:
        decision = await svc.decide_for_client(
            req.prompt,
            client=get_http_client(),
            expects_structured_output=req.expects_structured_output,
            profile=profile,
        )
        total_latency = (time.perf_counter() - started) * 1000.0
        _emit_telemetry(
            RoutingTelemetryEvent(
                request_id=decision.request_id,
                route=decision.classification.route,
                selected_model=decision.selected_model,
                selected_provider=f"passthrough:{profile.name}",
                tier=decision.tier.value,
                cache_hit=decision.cache_hit,
                classification_latency_ms=decision.classification_latency_ms,
                policy_latency_ms=0.0,
                provider_latency_ms=0.0,
                total_latency_ms=total_latency,
                estimated_cost_usd=None,
                actual_cost_usd=None,
                input_tokens=max(1, len(req.prompt) // 4),
                output_tokens=0,
                fallback_attempts=0,
                error_category=None,
                route_confidence=decision.classification.similarity_score,
                config_versions={
                    "policy_version": f"client-{profile.name}",
                    "route_version": decision.classification.route_config_version,
                    "classifier_version": decision.classification.classifier_version,
                },
                decision=None,
            )
        )
        payload = {
            "request_id": decision.request_id,
            "mode": "passthrough",
            "profile": profile.name,
            "model_key": decision.selected_model,
            "provider_model_id": decision.selected_model,
            "provider": f"passthrough:{profile.name}",
            "tier": decision.tier.value,
            "reason": decision.reason,
            "cache_hit": decision.cache_hit,
            "classification_confidence": decision.classification.similarity_score,
            "tier_models": dict(profile.models),
        }
        if req.include_diagnostics:
            payload["classification"] = asdict(decision.classification)
        return payload

    constraints = RoutingConstraints(
        max_cost_usd=req.max_cost_usd,
        max_latency_ms=req.max_latency_ms,
        min_context_window=req.min_context_window,
        require_structured_output=req.expects_structured_output,
        require_tool_calling=req.require_tool_calling,
    )
    route_result = svc.decide(req.prompt, constraints, expects_structured_output=req.expects_structured_output)
    decision = route_result.decision
    selected = decision.selected_model
    total_latency = (time.perf_counter() - started) * 1000.0
    _emit_telemetry(
        RoutingTelemetryEvent(
            request_id=route_result.request_id,
            route=decision.classification.route,
            selected_model=selected.key if selected else None,
            selected_provider=selected.provider.value if selected else None,
            tier=decision.classification.tier.value,
            cache_hit=route_result.cache_hit,
            classification_latency_ms=route_result.classification_latency_ms,
            policy_latency_ms=route_result.policy_latency_ms,
            provider_latency_ms=0.0,
            total_latency_ms=total_latency,
            estimated_cost_usd=next(
                (c.estimated_cost_usd for c in decision.candidates if selected and c.model.key == selected.key),
                None,
            ),
            actual_cost_usd=None,
            input_tokens=max(1, len(req.prompt) // 4),
            output_tokens=0,
            fallback_attempts=0,
            error_category=decision.error_category.value if decision.error_category else None,
            route_confidence=decision.classification.similarity_score,
            config_versions={
                "policy_version": decision.policy_version,
                "route_version": decision.classification.route_config_version,
                "classifier_version": decision.classification.classifier_version,
            },
            decision=decision,
        )
    )
    payload = {
        "request_id": route_result.request_id,
        "mode": "execute",
        "model_key": selected.key if selected else None,
        "provider_model_id": selected.provider_model_id if selected else None,
        "provider": selected.provider.value if selected else None,
        "tier": decision.classification.tier.value,
        "reason": decision.reason,
        "cache_hit": route_result.cache_hit,
        "classification_confidence": decision.classification.similarity_score,
    }
    if req.include_diagnostics:
        payload["classification"] = asdict(decision.classification)
        payload["candidates"] = [asdict(c) for c in decision.candidates]
    return payload


@app.post("/complete")
async def complete(req: CompleteRequest, _: None = Depends(require_router_auth)):
    svc = _service_or_raise()
    _validate_request_limits(svc, prompt=req.prompt, max_output_tokens=req.max_output_tokens)
    profile = svc.active_profile
    if profile is not None and profile.is_passthrough:
        endpoint = "/v1/messages" if profile.protocol == "anthropic" else "/v1/chat/completions"
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error_category="invalid_request",
                message=(
                    f"Passthrough profile '{profile.name}' cannot execute via /complete. "
                    f"Use {endpoint} so Dispatch can forward to your upstream."
                ),
            ).model_dump(),
        )
    constraints = RoutingConstraints(
        max_cost_usd=req.max_cost_usd,
        max_latency_ms=req.max_latency_ms,
        min_context_window=req.min_context_window,
        require_structured_output=req.expects_structured_output,
        require_tool_calling=req.require_tool_calling,
    )
    started = time.perf_counter()
    route_result, execution = await svc.complete(
        req.prompt,
        constraints,
        max_output_tokens=req.max_output_tokens,
        expects_structured_output=req.expects_structured_output,
    )
    total_latency = (time.perf_counter() - started) * 1000.0
    decision = route_result.decision
    selected = execution.used_model or decision.selected_model

    if execution.error is not None:
        _emit_telemetry(
            RoutingTelemetryEvent(
                request_id=route_result.request_id,
                route=decision.classification.route,
                selected_model=selected.key if selected else None,
                selected_provider=selected.provider.value if selected else None,
                tier=decision.classification.tier.value,
                cache_hit=route_result.cache_hit,
                classification_latency_ms=route_result.classification_latency_ms,
                policy_latency_ms=route_result.policy_latency_ms,
                provider_latency_ms=0.0,
                total_latency_ms=total_latency,
                estimated_cost_usd=None,
                actual_cost_usd=None,
                input_tokens=max(1, len(req.prompt) // 4),
                output_tokens=0,
                fallback_attempts=len(execution.fallback_attempts),
                error_category=execution.error.category.value,
                route_confidence=decision.classification.similarity_score,
                config_versions={
                    "policy_version": decision.policy_version,
                    "route_version": decision.classification.route_config_version,
                    "classifier_version": decision.classification.classifier_version,
                },
                decision=decision,
            )
        )
        status = 422 if execution.error.category.value == "no_feasible_model" else 502
        raise HTTPException(
            status_code=status,
            detail=ErrorResponse(
                error_category=execution.error.category.value,
                message=execution.error.message,
            ).model_dump(),
        )
    response = execution.response
    if response is None:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(error_category="internal_error", message="missing provider response").model_dump(),
        )
    estimated = next(
        (c.estimated_cost_usd for c in decision.candidates if selected and c.model.key == selected.key),
        None,
    )
    _emit_telemetry(
        RoutingTelemetryEvent(
            request_id=route_result.request_id,
            route=decision.classification.route,
            selected_model=selected.key if selected else None,
            selected_provider=response.provider.value,
            tier=decision.classification.tier.value,
            cache_hit=route_result.cache_hit,
            classification_latency_ms=route_result.classification_latency_ms,
            policy_latency_ms=route_result.policy_latency_ms,
            provider_latency_ms=response.latency_ms,
            total_latency_ms=total_latency,
            estimated_cost_usd=estimated,
            actual_cost_usd=execution.actual_cost_usd,
            input_tokens=response.usage_input_tokens,
            output_tokens=response.usage_output_tokens,
            fallback_attempts=len(execution.fallback_attempts),
            error_category=None,
            route_confidence=decision.classification.similarity_score,
            config_versions={
                "policy_version": decision.policy_version,
                "route_version": decision.classification.route_config_version,
                "classifier_version": decision.classification.classifier_version,
            },
            decision=decision,
        )
    )
    return {
        "request_id": route_result.request_id,
        "response": response.text,
        "model_key": selected.key if selected else None,
        "provider_model_id": response.model,
        "provider": response.provider.value,
        "tier": decision.classification.tier.value,
        "usage": {
            "input_tokens": response.usage_input_tokens,
            "output_tokens": response.usage_output_tokens,
        },
        "fallback_attempts": execution.fallback_attempts,
        "cost": {"estimated": estimated, "actual": execution.actual_cost_usd},
        "latency_ms": response.latency_ms,
    }
