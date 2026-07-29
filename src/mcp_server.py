from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mcp.server.mcpserver import MCPServer

from .router.bootstrap import build_routing_service
from .router.schemas import RoutingConstraints, Tier
from .router.service import RoutingService
from .router.surface import RequestLimitError, ready_payload, validate_request_limits

mcp = MCPServer("dispatch-router")

_service: RoutingService | None = None
_startup_error: str | None = None


def _ensure_service() -> None:
    global _service, _startup_error
    if _service is None and _startup_error is None:
        _service, _startup_error = build_routing_service(log_startup=True)


def _service_or_error() -> RoutingService:
    _ensure_service()
    if _service is None:
        raise RuntimeError(_startup_error or "service unavailable")
    return _service


def _tool_error(message: str, *, category: str = "invalid_request") -> dict[str, Any]:
    return {"error": {"category": category, "message": message, "retryable": False}}


@mcp.tool()
def health() -> dict[str, Any]:
    """Return router process health and active mode."""
    _ensure_service()
    if _service is None:
        return {
            "status": "error",
            "error": _startup_error or "service unavailable",
            "profile": None,
            "mode": None,
        }
    profile = _service.active_profile
    return {
        "status": "ok",
        "profile": profile.name if profile else None,
        "mode": profile.mode if profile else None,
    }


@mcp.tool()
def ready() -> dict[str, Any]:
    """Return initialization and model readiness details."""
    _ensure_service()
    if _service is None:
        return {"ready": False, "error": _startup_error or "service unavailable"}
    return ready_payload(_service)


@mcp.tool()
def route(
    prompt: str,
    max_cost_usd: float | None = None,
    max_latency_ms: float | None = None,
    min_context_window: int | None = None,
    expects_structured_output: bool = False,
    require_tool_calling: bool = False,
    include_diagnostics: bool = False,
) -> dict[str, Any]:
    """Classify prompt and choose model without executing provider call."""
    try:
        svc = _service_or_error()
        validate_request_limits(svc, prompt=prompt, max_output_tokens=1)
    except RuntimeError as exc:
        return _tool_error(str(exc), category="service_unavailable")
    except (RequestLimitError, ValueError) as exc:
        return _tool_error(str(exc), category="invalid_request")

    profile = svc.active_profile

    if profile is not None and profile.is_passthrough:
        try:
            decision = svc.decide_for_client(
                prompt,
                expects_structured_output=expects_structured_output,
                profile=profile,
            )
        except Exception as exc:
            return _tool_error(str(exc), category="classification_failed")
        payload: dict[str, Any] = {
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
        if include_diagnostics:
            payload["classification"] = asdict(decision.classification)
        return payload

    constraints = RoutingConstraints(
        max_cost_usd=max_cost_usd,
        max_latency_ms=max_latency_ms,
        min_context_window=min_context_window,
        require_structured_output=expects_structured_output,
        require_tool_calling=require_tool_calling,
    )
    route_result = svc.decide(prompt, constraints, expects_structured_output=expects_structured_output)
    decision = route_result.decision
    selected = decision.selected_model
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
    if include_diagnostics:
        payload["classification"] = asdict(decision.classification)
        payload["candidates"] = [asdict(c) for c in decision.candidates]
    return payload


@mcp.tool()
def complete(
    prompt: str,
    max_output_tokens: int = 512,
    max_cost_usd: float | None = None,
    max_latency_ms: float | None = None,
    min_context_window: int | None = None,
    expects_structured_output: bool = False,
    require_tool_calling: bool = False,
) -> dict[str, Any]:
    """Classify, choose model, execute provider call, and return response."""
    try:
        svc = _service_or_error()
        validate_request_limits(svc, prompt=prompt, max_output_tokens=max_output_tokens)
    except RuntimeError as exc:
        return _tool_error(str(exc), category="service_unavailable")
    except (RequestLimitError, ValueError) as exc:
        return _tool_error(str(exc), category="invalid_request")

    profile = svc.active_profile
    if profile is not None and profile.is_passthrough:
        endpoint = "/v1/messages" if profile.protocol == "anthropic" else "/v1/chat/completions"
        return _tool_error(
            f"Passthrough profile '{profile.name}' cannot execute via MCP complete. "
            f"Use the HTTP API {endpoint} so Dispatch can forward to your upstream, "
            "or switch DISPATCH_PROFILE=demo for execute mode.",
            category="invalid_request",
        )

    constraints = RoutingConstraints(
        max_cost_usd=max_cost_usd,
        max_latency_ms=max_latency_ms,
        min_context_window=min_context_window,
        require_structured_output=expects_structured_output,
        require_tool_calling=require_tool_calling,
    )
    route_result, execution = svc.complete(
        prompt,
        constraints,
        max_output_tokens=max_output_tokens,
        expects_structured_output=expects_structured_output,
    )
    decision = route_result.decision
    selected = execution.used_model or decision.selected_model

    if execution.error is not None:
        return {
            "request_id": route_result.request_id,
            "error": {
                "category": execution.error.category.value,
                "message": execution.error.message,
                "retryable": execution.error.retryable,
            },
            "fallback_attempts": execution.fallback_attempts,
        }

    response = execution.response
    if response is None:
        return {
            "request_id": route_result.request_id,
            "error": {
                "category": "internal_error",
                "message": "missing provider response",
                "retryable": False,
            },
            "fallback_attempts": execution.fallback_attempts,
        }

    estimated = next(
        (c.estimated_cost_usd for c in decision.candidates if selected and c.model.key == selected.key),
        None,
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


@mcp.tool()
def refresh_models() -> dict[str, Any]:
    """Refresh live model catalogs and return updated tier mappings."""
    try:
        svc = _service_or_error()
    except RuntimeError as exc:
        return _tool_error(str(exc), category="service_unavailable")
    refreshed = svc.model_registry.refresh()
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


def main() -> None:
    _ensure_service()
    if _service is None:
        raise SystemExit(f"dispatch-mcp failed to start: {_startup_error}")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
