"""OpenAI-compatible API shim for Codex / OpenCode / Cursor / Claude Code.

Two modes (from configs/clients.yaml / DISPATCH_* env):

  execute     — Dispatch classifies + calls its own providers (demo).
  passthrough — Dispatch classifies, maps tier → YOUR model, forwards the
                full request (tools, stream, etc.) to YOUR upstream with
                YOUR API key. Dispatch is the router, not the model.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .router.prompting import build_routing_prompt
from .router.passthrough import (
    dispatch_headers,
    forward_openai_chat,
    has_upstream_credentials,
    resolve_auth_headers,
)
from .router.schemas import RoutingConstraints
from .router.security import require_router_auth, verify_router_api_key
from .router.service import RoutingService
from .router.telemetry import RoutingTelemetryEvent


openai_router = APIRouter(prefix="/v1", tags=["openai-compatible"])


class ChatMessage(BaseModel):
    role: str
    content: Any = ""
    name: str | None = None
    tool_calls: Any | None = None
    tool_call_id: str | None = None


class ChatCompletionsRequest(BaseModel):
    model: str = "dispatch"
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float | None = 0.0
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    user: str | None = None
    tools: Any | None = None
    tool_choice: Any | None = None
    response_format: Any | None = None
    # Optional router extensions (ignored by most clients)
    max_cost_usd: float | None = None
    max_latency_ms: float | None = None
    min_context_window: int | None = None
    expects_structured_output: bool = False
    require_tool_calling: bool = False

    model_config = {"extra": "allow"}


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
        return "\n".join(p for p in parts if p)
    return str(content)


def messages_to_prompt(messages: list[ChatMessage], *, max_chars: int = 4000) -> str:
    """Build stable classification text from latest user intent."""
    normalized: list[tuple[str, str]] = []
    for message in messages:
        text = _message_text(message.content).strip()
        if not text:
            continue
        role = message.role if message.role in {"system", "user", "assistant", "human"} else "user"
        normalized.append((role, text))
    return build_routing_prompt(normalized, max_chars=max_chars)


def normalize_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        text = _message_text(message.content)
        role = message.role if message.role in {"system", "user", "assistant"} else "user"
        normalized.append({"role": role, "content": text})
    return normalized


def openai_error(message: str, *, err_type: str = "invalid_request_error", code: str | None = None, status: int = 400):
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": err_type,
                "param": None,
                "code": code,
            }
        },
    )


def check_api_key(authorization: str | None) -> None:
    """Back-compat wrapper used by unit tests."""
    verify_router_api_key(authorization)


def build_chat_completion_response(
    *,
    completion_id: str,
    model: str,
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
    created: int,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def iter_sse_chunks(*, completion_id: str, model: str, text: str, created: int) -> Iterator[str]:
    """Pseudo-stream: one role chunk, one content chunk, then stop + DONE."""
    first = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first)}\n\n"

    chunk_size = 48
    for i in range(0, max(len(text), 1), chunk_size):
        piece = text[i : i + chunk_size] if text else ""
        mid = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(mid)}\n\n"

    last = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(last)}\n\n"
    yield "data: [DONE]\n\n"


def _raw_payload(req: ChatCompletionsRequest) -> dict[str, Any]:
    """Preserve extra OpenAI fields (tools, etc.) for upstream passthrough."""
    data = req.model_dump(exclude_none=True)
    # Router-only fields should not be sent upstream.
    for key in (
        "max_cost_usd",
        "max_latency_ms",
        "min_context_window",
        "expects_structured_output",
        "require_tool_calling",
    ):
        data.pop(key, None)
    return data


def _response_format_requires_structured_output(response_format: Any) -> bool:
    """Only JSON response formats require structured-output-capable models."""
    if response_format is None:
        return False
    if isinstance(response_format, str):
        normalized = response_format.strip().lower()
        return normalized in {"json_object", "json_schema"}
    if isinstance(response_format, dict):
        kind = str(response_format.get("type", "")).strip().lower()
        return kind in {"json_object", "json_schema"}
    return bool(response_format)


def register_openai_routes(
    *,
    get_service,
    emit_telemetry,
) -> APIRouter:
    @openai_router.get("/models")
    def list_models(_: None = Depends(require_router_auth)):
        svc: RoutingService = get_service()
        now = int(time.time())
        data = [
            {
                "id": "dispatch",
                "object": "model",
                "created": now,
                "owned_by": "dispatch",
            }
        ]
        profile = svc.active_profile
        if profile and profile.is_passthrough:
            for tier, model_id in profile.models.items():
                if not model_id:
                    continue
                data.append(
                    {
                        "id": model_id,
                        "object": "model",
                        "created": now,
                        "owned_by": f"dispatch:{profile.name}:{tier}",
                    }
                )
        else:
            for model in svc.model_registry.enabled_models():
                data.append(
                    {
                        "id": model.key,
                        "object": "model",
                        "created": now,
                        "owned_by": model.provider.value,
                    }
                )
                data.append(
                    {
                        "id": model.provider_model_id,
                        "object": "model",
                        "created": now,
                        "owned_by": model.provider.value,
                    }
                )
        return {"object": "list", "data": data}

    @openai_router.post("/chat/completions")
    async def chat_completions(
        req: ChatCompletionsRequest,
        request: Request,
        _: None = Depends(require_router_auth),
    ):
        svc: RoutingService = get_service()
        profile = svc.active_profile

        prompt = messages_to_prompt(req.messages)
        if not prompt:
            return openai_error("messages must contain non-empty content")

        if len(prompt) > svc.config.request_max_prompt_chars:
            return openai_error("prompt exceeds configured limit", status=400)

        if req.max_tokens is not None:
            max_tokens = req.max_tokens
        elif req.max_completion_tokens is not None:
            max_tokens = req.max_completion_tokens
        else:
            max_tokens = 1024
        if max_tokens <= 0:
            return openai_error("max_tokens must be > 0", status=400)
        if max_tokens > svc.config.request_max_output_tokens:
            return openai_error("max_tokens exceeds configured limit", status=400)

        # --- Passthrough: route with client's models ---
        if profile is not None and profile.is_passthrough:
            return await _passthrough_chat(
                svc=svc,
                profile=profile,
                req=req,
                request=request,
                prompt=prompt,
                emit_telemetry=emit_telemetry,
            )

        # --- Execute: Dispatch's own providers ---
        return _execute_chat(
            svc=svc,
            req=req,
            prompt=prompt,
            max_tokens=max_tokens,
            emit_telemetry=emit_telemetry,
        )

    return openai_router


async def _passthrough_chat(
    *,
    svc: RoutingService,
    profile,
    req: ChatCompletionsRequest,
    request: Request,
    prompt: str,
    emit_telemetry,
):
    started = time.perf_counter()
    try:
        # Prefer forwarded tool key for discovery + upstream.
        incoming_key = None
        auth_header = request.headers.get("authorization") or ""
        if auth_header.lower().startswith("bearer "):
            incoming_key = auth_header[7:].strip()
        incoming_key = incoming_key or request.headers.get("x-api-key")
        decision = svc.decide_for_client(
            prompt,
            expects_structured_output=(
                req.expects_structured_output
                or _response_format_requires_structured_output(req.response_format)
            ),
            profile=profile,
            upstream_api_key=profile.upstream_api_key or incoming_key,
        )
    except Exception as exc:
        return openai_error(str(exc), err_type="server_error", status=500)

    auth = resolve_auth_headers(
        profile,
        incoming_authorization=request.headers.get("authorization"),
        incoming_api_key=request.headers.get("x-api-key"),
        incoming_auth_token=None,
        protocol="openai",
    )
    # When ROUTER_API_KEY is set, the client Bearer authenticates Dispatch —
    # never forward it upstream. Require a dedicated upstream key instead.
    from .router.security import router_api_key, router_api_key_configured

    if router_api_key_configured():
        expected = router_api_key()
        incoming = (request.headers.get("authorization") or "").removeprefix("Bearer ").strip()
        if expected and incoming == expected:
            if profile.upstream_api_key:
                auth = {"Authorization": f"Bearer {profile.upstream_api_key}"}
            else:
                return openai_error(
                    "Passthrough needs DISPATCH_UPSTREAM_API_KEY (or upstream_api_key in "
                    "clients.yaml) when ROUTER_API_KEY is set — the tool's Bearer is for "
                    "Dispatch, not upstream.",
                    err_type="authentication_error",
                    status=401,
                )

    if not has_upstream_credentials(auth, protocol="openai"):
        return openai_error(
            "Passthrough needs an upstream API key (Authorization Bearer, x-api-key, "
            "or DISPATCH_UPSTREAM_API_KEY).",
            err_type="authentication_error",
            status=401,
        )

    try:
        result = forward_openai_chat(
            profile=profile,
            selected_model=decision.selected_model,
            payload=_raw_payload(req),
            auth_headers=auth,
            request_id=decision.request_id,
            tier=decision.tier,
        )
    except Exception as exc:
        return openai_error(f"upstream forward failed: {exc}", err_type="upstream_error", status=502)

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
            input_tokens=max(1, len(prompt) // 4),
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

    # Annotate non-stream JSON with dispatch metadata when possible.
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
                "cache_hit": decision.cache_hit,
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


def _execute_chat(*, svc: RoutingService, req: ChatCompletionsRequest, prompt: str, max_tokens: int, emit_telemetry):
    has_tools_hint = (
        req.require_tool_calling
        or req.tools is not None
        or any(
            isinstance(m.content, list)
            and any(isinstance(p, dict) and p.get("type") == "tool_use" for p in m.content)
            for m in req.messages
        )
    )
    constraints = RoutingConstraints(
        max_cost_usd=req.max_cost_usd,
        max_latency_ms=req.max_latency_ms,
        min_context_window=req.min_context_window,
        require_structured_output=(
            req.expects_structured_output
            or _response_format_requires_structured_output(req.response_format)
        ),
        require_tool_calling=has_tools_hint,
        estimated_output_tokens=max_tokens,
    )

    started = time.perf_counter()
    route_result, execution = svc.complete(
        prompt,
        constraints,
        max_output_tokens=max_tokens,
        expects_structured_output=constraints.require_structured_output,
        messages=normalize_messages(req.messages),
        temperature=float(req.temperature or 0.0),
    )
    total_latency = (time.perf_counter() - started) * 1000.0
    decision = route_result.decision
    selected = execution.used_model or decision.selected_model

    if execution.error is not None:
        emit_telemetry(
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
                input_tokens=max(1, len(prompt) // 4),
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
        return openai_error(
            execution.error.message,
            err_type="upstream_error",
            code=execution.error.category.value,
            status=status,
        )

    response = execution.response
    if response is None:
        return openai_error("missing provider response", err_type="server_error", status=500)

    estimated = next(
        (c.estimated_cost_usd for c in decision.candidates if selected and c.model.key == selected.key),
        None,
    )
    emit_telemetry(
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

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model_name = selected.provider_model_id if selected else req.model
    headers = {
        "X-Dispatch-Request-Id": route_result.request_id,
        "X-Dispatch-Tier": decision.classification.tier.value,
        "X-Dispatch-Provider": response.provider.value,
        "X-Dispatch-Model-Key": selected.key if selected else "",
        "X-Dispatch-Mode": "execute",
        "X-Dispatch-Cache-Hit": "1" if route_result.cache_hit else "0",
    }

    if req.stream:
        return StreamingResponse(
            iter_sse_chunks(
                completion_id=completion_id,
                model=model_name,
                text=response.text,
                created=created,
            ),
            media_type="text/event-stream",
            headers=headers,
        )

    body = build_chat_completion_response(
        completion_id=completion_id,
        model=model_name,
        text=response.text,
        prompt_tokens=response.usage_input_tokens,
        completion_tokens=response.usage_output_tokens,
        created=created,
    )
    body["dispatch"] = {
        "request_id": route_result.request_id,
        "tier": decision.classification.tier.value,
        "provider": response.provider.value,
        "model_key": selected.key if selected else None,
        "reason": decision.reason,
        "cache_hit": route_result.cache_hit,
        "cost_usd": execution.actual_cost_usd,
        "mode": "execute",
    }
    return JSONResponse(content=body, headers=headers)
