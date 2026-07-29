"""Shared helpers for HTTP API and MCP surfaces."""

from __future__ import annotations

from typing import Any

from .service import RoutingService


class RequestLimitError(ValueError):
    """Invalid prompt / token limits for a routing request."""


def validate_request_limits(
    svc: RoutingService,
    *,
    prompt: str,
    max_output_tokens: int,
) -> None:
    if not prompt or not str(prompt).strip():
        raise RequestLimitError("prompt must not be blank")
    if len(prompt) > svc.config.request_max_prompt_chars:
        raise RequestLimitError("prompt exceeds configured maximum length")
    if max_output_tokens <= 0:
        raise RequestLimitError("max_output_tokens must be > 0")
    if max_output_tokens > svc.config.request_max_output_tokens:
        raise RequestLimitError("max_output_tokens exceeds configured maximum")


def ready_payload(svc: RoutingService) -> dict[str, Any]:
    profile = svc.active_profile
    registry = svc.model_registry
    payload: dict[str, Any] = {
        "ready": True,
        "models_source": registry.source,
        "models_version": registry.version,
        "models": [
            {
                "key": m.key,
                "id": m.provider_model_id,
                "provider": m.provider.value,
                "tier": m.tier.value,
            }
            for m in registry.enabled_models()
        ],
    }
    if profile is not None:
        payload["profile"] = profile.name
        payload["mode"] = profile.mode
        payload["protocol"] = profile.protocol
        if profile.is_passthrough:
            try:
                profile.ensure_models(api_key=profile.upstream_api_key or None)
            except Exception as exc:
                payload["discover_error"] = str(exc)
            payload["tier_models"] = dict(profile.models)
            payload["upstream"] = profile.upstream_base_url
    return payload
