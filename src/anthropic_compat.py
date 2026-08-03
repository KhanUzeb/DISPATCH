"""Anthropic Messages API shim (`/v1/messages`).

Point an Anthropic-compatible client at Dispatch with:

  ANTHROPIC_BASE_URL=http://localhost:8000
  ANTHROPIC_AUTH_TOKEN=<your anthropic or gateway key>

Dispatch classifies the prompt, rewrites `model` to the mapped tier model from
the `anthropic` client profile, and forwards to Anthropic (or a compatible
gateway).
"""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .router.prompting import build_routing_prompt
from .router.passthrough import (
    dispatch_headers,
    forward_anthropic_messages,
    has_upstream_credentials,
    resolve_auth_headers,
)
from .router.security import require_router_auth, router_api_key, router_api_key_configured
from .router.service import RoutingService
from .router.telemetry import RoutingTelemetryEvent
from .router.tokens import estimate_input_tokens


anthropic_router = APIRouter(prefix="/v1", tags=["anthropic-compatible"])


class AnthropicMessage(BaseModel):
    role: str
    content: Any = ""

    model_config = {"extra": "allow"}


class MessagesRequest(BaseModel):
    model: str = "dispatch"
    messages: list[AnthropicMessage] = Field(..., min_length=1)
    max_tokens: int = 32768
    system: Any | None = None
    temperature: float | None = None
    stream: bool = False
    tools: Any | None = None
    tool_choice: Any | None = None
    metadata: Any | None = None
    stop_sequences: list[str] | None = None
    top_p: float | None = None
    top_k: int | None = None

    model_config = {"extra": "allow"}


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif "text" in block:
                    parts.append(str(block["text"]))
        return "\n".join(p for p in parts if p)
    return str(content)


def anthropic_messages_to_prompt(req: MessagesRequest, *, max_chars: int = 4000) -> str:
    normalized: list[tuple[str, str]] = []
    if req.system:
        sys_text = _content_text(req.system).strip()
        if sys_text:
            normalized.append(("system", sys_text))
    for message in req.messages:
        text = _content_text(message.content).strip()
        if not text:
            continue
        role = message.role if message.role in {"user", "assistant"} else "user"
        normalized.append((role, text))
    return build_routing_prompt(normalized, max_chars=max_chars)


def anthropic_error(message: str, *, err_type: str = "invalid_request_error", status: int = 400):
    return JSONResponse(
        status_code=status,
        content={"type": "error", "error": {"type": err_type, "message": message}},
    )


def register_anthropic_routes(*, get_service, emit_telemetry) -> APIRouter:
    @anthropic_router.post("/messages")
    async def messages(
        req: MessagesRequest,
        request: Request,
        _: None = Depends(require_router_auth),
    ):
        svc: RoutingService = get_service()
        profile = svc.active_profile

        prompt = anthropic_messages_to_prompt(req)
        if not prompt:
            return anthropic_error("messages must contain non-empty content")
        if len(prompt) > svc.config.request_max_prompt_chars:
            return anthropic_error("prompt exceeds configured limit")
        if req.max_tokens <= 0:
            return anthropic_error("max_tokens must be > 0")
        if req.max_tokens > svc.config.request_max_output_tokens:
            return anthropic_error("max_tokens exceeds configured limit")

        if profile is None or not profile.is_passthrough:
            return anthropic_error(
                "Anthropic /v1/messages requires passthrough mode. "
                "Set DISPATCH_PROFILE=anthropic (or another anthropic profile).",
                err_type="invalid_request_error",
                status=400,
            )
        if profile.protocol != "anthropic":
            return anthropic_error(
                f"Active profile '{profile.name}' protocol is '{profile.protocol}', "
                "expected 'anthropic'. Set DISPATCH_PROFILE=anthropic.",
                status=400,
            )

        started = time.perf_counter()
        try:
            incoming_key = request.headers.get("x-api-key")
            auth_header = request.headers.get("authorization") or ""
            if auth_header.lower().startswith("bearer "):
                incoming_key = incoming_key or auth_header[7:].strip()
            decision = await svc.decide_for_client(
                prompt,
                client=request.app.state.http_client,
                profile=profile,
                upstream_api_key=profile.upstream_api_key or incoming_key,
            )
        except Exception as exc:
            return anthropic_error(str(exc), err_type="api_error", status=500)

        auth = resolve_auth_headers(
            profile,
            incoming_authorization=request.headers.get("authorization"),
            incoming_api_key=request.headers.get("x-api-key"),
            incoming_auth_token=None,
            protocol="anthropic",
        )
        if router_api_key_configured():
            expected = router_api_key()
            bearer = (request.headers.get("authorization") or "").removeprefix("Bearer ").strip()
            x_key = (request.headers.get("x-api-key") or "").strip()
            if expected and (bearer == expected or x_key == expected):
                if not profile.upstream_api_key:
                    return anthropic_error(
                        "Set DISPATCH_UPSTREAM_API_KEY for Anthropic upstream when "
                        "ROUTER_API_KEY is configured.",
                        err_type="authentication_error",
                        status=401,
                    )
                key = profile.upstream_api_key
                if key.startswith("sk-ant-"):
                    auth = {"x-api-key": key, "anthropic-version": "2023-06-01"}
                else:
                    auth = {
                        "Authorization": f"Bearer {key}",
                        "anthropic-version": "2023-06-01",
                    }

        # Preserve Anthropic beta headers if the client sent them.
        for header_name in ("anthropic-version", "anthropic-beta"):
            value = request.headers.get(header_name)
            if value:
                auth[header_name] = value
        auth.setdefault("anthropic-version", "2023-06-01")

        if not has_upstream_credentials(auth, protocol="anthropic"):
            return anthropic_error(
                "Passthrough needs an Anthropic upstream key (x-api-key, Authorization, "
                "or DISPATCH_UPSTREAM_API_KEY).",
                err_type="authentication_error",
                status=401,
            )

        payload = req.model_dump(exclude_none=True)
        try:
            result = await forward_anthropic_messages(
                profile=profile,
                selected_model=decision.selected_model,
                payload=payload,
                auth_headers=auth,
                client=request.app.state.http_client,
                request_id=decision.request_id,
                tier=decision.tier,
            )
        except Exception as exc:
            return anthropic_error(f"upstream forward failed: {exc}", err_type="api_error", status=502)

        total_latency = (time.perf_counter() - started) * 1000.0
        emit_telemetry(
            RoutingTelemetryEvent(
                request_id=decision.request_id,
                route=decision.classification.route,
                selected_model=decision.selected_model,
                selected_provider=f"passthrough:{profile.name}",
                tier=decision.tier.value,
                cache_hit=decision.cache_hit,
                classification_latency_ms=decision.classification_latency_ms,
                policy_latency_ms=0.0,
                provider_latency_ms=result.latency_ms,
                total_latency_ms=total_latency,
                estimated_cost_usd=None,
                actual_cost_usd=None,
                input_tokens=estimate_input_tokens(prompt),
                output_tokens=0,
                fallback_attempts=0,
                error_category=None if result.status_code < 400 else "upstream_error",
                route_confidence=decision.classification.similarity_score,
                config_versions={
                    "policy_version": f"client-{profile.name}",
                    "route_version": decision.classification.route_config_version,
                    "classifier_version": decision.classification.classifier_version,
                },
                decision=None,
            )
        )

        headers = dispatch_headers(
            request_id=decision.request_id,
            tier=decision.tier,
            selected_model=decision.selected_model,
            profile_name=profile.name,
            cache_hit=decision.cache_hit,
        )

        if result.stream is not None:
            return StreamingResponse(
                result.stream,
                status_code=result.status_code,
                media_type=result.headers.get("content-type", "text/event-stream"),
                headers=headers,
            )

        body = result.body or b"{}"
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict) and result.status_code < 400:
                parsed["dispatch"] = {
                    "request_id": decision.request_id,
                    "tier": decision.tier.value,
                    "model": decision.selected_model,
                    "profile": profile.name,
                    "mode": "passthrough",
                    "reason": decision.reason,
                }
                body = json.dumps(parsed).encode("utf-8")
        except Exception:
            pass

        return Response(
            content=body,
            status_code=result.status_code,
            media_type="application/json",
            headers=headers,
        )

    return anthropic_router
