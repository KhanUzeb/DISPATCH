from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import httpx

from ..schemas import Provider, ProviderRequest, ProviderResponse
from .base import ProviderAdapter
from .base import ProviderStreamChunk
from ..http_client import get_http_client


def _parse_sse_data_line(line: str) -> str | None:
    """Return the payload after ``data:``, or None if not a data line."""
    if not line.startswith("data:"):
        return None
    return line[5:].lstrip()


def _delta_text_from_chunk(payload: str) -> str | None:
    """Extract content delta from an OpenAI chat.completion.chunk JSON payload.

    Returns ``None`` for ``[DONE]`` / empty terminal markers; ``""`` for
    non-content chunks (role-only, finish, etc.).
    """
    if payload in ("", "[DONE]"):
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ""
    choices = data.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) else ""


class OpenAICompatibleAdapter(ProviderAdapter):
    """HTTP chat-completions adapter for OpenAI and OpenAI-compatible APIs."""

    def __init__(
        self,
        provider: Provider,
        api_key: str | None,
        base_url: str,
        timeout_ms: int = 30000,
        extra_headers: dict[str, str] | None = None,
        require_api_key: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_ms = timeout_ms
        self.extra_headers = extra_headers or {}
        self.require_api_key = require_api_key
        self.client = client

    @property
    def provider_name(self) -> str:
        return self.provider.value

    def _headers(self) -> dict[str, str]:
        if self.require_api_key and not self.api_key:
            raise RuntimeError(f"Missing API key for provider {self.provider.value}")
        headers = {
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _chat_payload(self, model_id: str, request: ProviderRequest, *, stream: bool) -> dict:
        payload = {
            "model": model_id,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        if stream:
            payload["stream"] = True
        return payload

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code == 401:
            raise RuntimeError(f"{self.provider.value} authentication failed")
        if response.status_code == 429:
            raise RuntimeError(f"{self.provider.value} rate limited")
        response.raise_for_status()

    async def execute(self, model_id: str, request: ProviderRequest) -> ProviderResponse:
        headers = self._headers()
        start = time.perf_counter()
        client = self.client or get_http_client()
        response = await client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=self._chat_payload(model_id, request, stream=False),
            timeout=self.timeout_ms / 1000.0,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        self._raise_for_status(response)
        data = response.json()
        usage = data.get("usage", {}) or {}
        return ProviderResponse(
            text=(data.get("choices", [{}])[0].get("message", {}).get("content") or ""),
            usage_input_tokens=int(usage.get("prompt_tokens", 0)),
            usage_output_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency_ms,
            model=model_id,
            provider=self.provider,
            provider_request_id=data.get("id"),
            rate_limited=False,
        )

    async def stream(self, model_id: str, request: ProviderRequest) -> AsyncIterator[str]:
        async for chunk in self.stream_with_usage(model_id, request):
            if chunk.text:
                yield chunk.text

    async def stream_with_usage(self, model_id: str, request: ProviderRequest) -> AsyncIterator[ProviderStreamChunk]:
        headers = self._headers()
        client = self.client or get_http_client()
        async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={**self._chat_payload(model_id, request, stream=True), "stream_options": {"include_usage": True}},
                timeout=self.timeout_ms / 1000.0,
        ) as response:
            self._raise_for_status(response)
            async for line in response.aiter_lines():
                if not line:
                    continue
                payload = _parse_sse_data_line(line)
                if payload is None or payload == "[DONE]":
                    continue
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                usage = data.get("usage") or {}
                text = _delta_text_from_chunk(payload) or ""
                yield ProviderStreamChunk(
                    text=text,
                    usage_input_tokens=int(usage["prompt_tokens"]) if "prompt_tokens" in usage else None,
                    usage_output_tokens=int(usage["completion_tokens"]) if "completion_tokens" in usage else None,
                )
