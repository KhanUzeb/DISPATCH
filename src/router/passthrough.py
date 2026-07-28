"""Upstream passthrough — forward routed requests to the client's provider."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Iterator

import httpx

from .clients import ClientProfile
from .schemas import Tier


@dataclass(frozen=True)
class PassthroughResult:
    status_code: int
    headers: dict[str, str]
    body: bytes | None
    stream: Iterator[bytes] | None
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
    """Prefer profile static key; else forward what the coding tool sent."""
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


def forward_openai_chat(
    *,
    profile: ClientProfile,
    selected_model: str,
    payload: dict[str, Any],
    auth_headers: dict[str, str],
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
    url = _openai_chat_url(profile.upstream_base_url)
    headers = {
        "Content-Type": "application/json",
        **auth_headers,
    }
    stream = bool(body.get("stream"))
    started = time.perf_counter()

    if stream:
        client = httpx.Client(timeout=timeout_s)
        req = client.build_request("POST", url, headers=headers, content=json.dumps(body))
        response = client.send(req, stream=True)
        latency_ms = (time.perf_counter() - started) * 1000.0

        def _iter() -> Iterator[bytes]:
            try:
                for chunk in response.iter_bytes():
                    if chunk:
                        yield chunk
            finally:
                response.close()
                client.close()

        return PassthroughResult(
            status_code=response.status_code,
            headers={k: v for k, v in response.headers.items()},
            body=None,
            stream=_iter(),
            selected_model=selected_model,
            tier=tier,
            latency_ms=latency_ms,
            request_id=request_id,
        )

    with httpx.Client(timeout=timeout_s) as client:
        response = client.post(url, headers=headers, content=json.dumps(body))
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


def forward_anthropic_messages(
    *,
    profile: ClientProfile,
    selected_model: str,
    payload: dict[str, Any],
    auth_headers: dict[str, str],
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
    url = _anthropic_messages_url(profile.upstream_base_url)
    headers = {
        "Content-Type": "application/json",
        **auth_headers,
    }
    stream = bool(body.get("stream"))
    started = time.perf_counter()

    if stream:
        client = httpx.Client(timeout=timeout_s)
        req = client.build_request("POST", url, headers=headers, content=json.dumps(body))
        response = client.send(req, stream=True)
        latency_ms = (time.perf_counter() - started) * 1000.0

        def _iter() -> Iterator[bytes]:
            try:
                for chunk in response.iter_bytes():
                    if chunk:
                        yield chunk
            finally:
                response.close()
                client.close()

        return PassthroughResult(
            status_code=response.status_code,
            headers={k: v for k, v in response.headers.items()},
            body=None,
            stream=_iter(),
            selected_model=selected_model,
            tier=tier,
            latency_ms=latency_ms,
            request_id=request_id,
        )

    with httpx.Client(timeout=timeout_s) as client:
        response = client.post(url, headers=headers, content=json.dumps(body))
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

