"""Upstream passthrough — forward routed requests to the client's provider."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from .clients import ClientProfile
from .schemas import Tier


@dataclass(frozen=True)
class PassthroughResult:
    status_code: int
    headers: dict[str, str]
    body: bytes | None
    stream: AsyncIterator[bytes] | None
    selected_model: str
    tier: Tier
    latency_ms: float
    request_id: str


def _openai_chat_url(base: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _anthropic_messages_url(base: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/messages"):
        return base
    if base.endswith("/v1"):
        return f"{base}/messages"
    return f"{base}/v1/messages"


def resolve_auth_headers(
    profile: ClientProfile,
    *,
    incoming_authorization: str | None,
    incoming_api_key: str | None,
    incoming_auth_token: str | None,
    protocol: str,
) -> dict[str, str]:
    """Prefer profile static key; else forward what the client sent."""
    headers: dict[str, str] = {}
    static = (profile.upstream_api_key or "").strip()

    if protocol == "anthropic":
        if static:
            # Anthropic first-party uses x-api-key; gateways often want Bearer.
            if static.startswith("sk-ant-"):
                headers["x-api-key"] = static
            else:
                headers["Authorization"] = f"Bearer {static}"
        else:
            if incoming_api_key:
                headers["x-api-key"] = incoming_api_key
            if incoming_auth_token:
                headers["Authorization"] = f"Bearer {incoming_auth_token}"
            elif incoming_authorization:
                headers["Authorization"] = incoming_authorization
        headers.setdefault("anthropic-version", "2023-06-01")
        return headers

    # OpenAI-compatible
    if static:
        headers["Authorization"] = f"Bearer {static}"
    elif incoming_authorization:
        headers["Authorization"] = incoming_authorization
    elif incoming_api_key:
        headers["Authorization"] = f"Bearer {incoming_api_key}"
    return headers


def has_upstream_credentials(auth_headers: dict[str, str], *, protocol: str) -> bool:
    """True when headers contain a usable upstream credential (not just metadata)."""
    if (auth_headers.get("x-api-key") or "").strip():
        return True
    auth = (auth_headers.get("Authorization") or "").strip()
    if not auth:
        return False
    if protocol == "anthropic":
        return True
    return auth.lower().startswith("bearer ") and len(auth) > len("Bearer ")


async def _forward(
    *,
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout_s: float,
    selected_model: str,
    request_id: str,
    tier: Tier,
) -> PassthroughResult:
    stream = bool(body.get("stream"))
    started = time.perf_counter()
    content = json.dumps(body)

    if stream:
        req = client.build_request("POST", url, headers=headers, content=content, timeout=timeout_s)
        response = await client.send(req, stream=True)
        latency_ms = (time.perf_counter() - started) * 1000.0

        async def _aiter() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
            finally:
                await response.aclose()

        return PassthroughResult(
            status_code=response.status_code,
            headers={k: v for k, v in response.headers.items()},
            body=None,
            stream=_aiter(),
            selected_model=selected_model,
            tier=tier,
            latency_ms=latency_ms,
            request_id=request_id,
        )

    response = await client.post(url, headers=headers, content=content, timeout=timeout_s)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return PassthroughResult(
        status_code=response.status_code,
        headers={k: v for k, v in response.headers.items()},
        body=response.content,
        stream=None,
        selected_model=selected_model,
        tier=tier,
        latency_ms=latency_ms,
        request_id=request_id,
    )


async def forward_openai_chat(
    *,
    profile: ClientProfile,
    selected_model: str,
    payload: dict[str, Any],
    auth_headers: dict[str, str],
    client: httpx.AsyncClient,
    timeout_s: float = 120.0,
    request_id: str,
    tier: Tier,
) -> PassthroughResult:
    if not profile.upstream_base_url:
        raise ValueError(
            "Passthrough profile missing upstream_base_url "
            "(set DISPATCH_UPSTREAM_BASE_URL or clients.yaml)"
        )

    body = dict(payload)
    body["model"] = selected_model
    return await _forward(
        client=client,
        url=_openai_chat_url(profile.upstream_base_url),
        headers={"Content-Type": "application/json", **auth_headers},
        body=body,
        timeout_s=timeout_s,
        selected_model=selected_model,
        request_id=request_id,
        tier=tier,
    )


async def forward_anthropic_messages(
    *,
    profile: ClientProfile,
    selected_model: str,
    payload: dict[str, Any],
    auth_headers: dict[str, str],
    client: httpx.AsyncClient,
    timeout_s: float = 120.0,
    request_id: str,
    tier: Tier,
) -> PassthroughResult:
    if not profile.upstream_base_url:
        raise ValueError(
            "Passthrough profile missing upstream_base_url "
            "(set DISPATCH_UPSTREAM_BASE_URL or clients.yaml)"
        )

    body = dict(payload)
    body["model"] = selected_model
    return await _forward(
        client=client,
        url=_anthropic_messages_url(profile.upstream_base_url),
        headers={"Content-Type": "application/json", **auth_headers},
        body=body,
        timeout_s=timeout_s,
        selected_model=selected_model,
        request_id=request_id,
        tier=tier,
    )


def dispatch_headers(
    *,
    request_id: str,
    tier: Tier,
    selected_model: str,
    profile_name: str,
    cache_hit: bool,
) -> dict[str, str]:
    return {
        "X-Dispatch-Request-Id": request_id,
        "X-Dispatch-Tier": tier.value,
        "X-Dispatch-Model": selected_model,
        "X-Dispatch-Profile": profile_name,
        "X-Dispatch-Mode": "passthrough",
        "X-Dispatch-Cache-Hit": "1" if cache_hit else "0",
    }
