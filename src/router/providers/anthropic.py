from __future__ import annotations

import time

import httpx

from ..schemas import Provider, ProviderRequest, ProviderResponse
from .base import ProviderAdapter
from ..http_client import get_http_client


class AnthropicAdapter(ProviderAdapter):
    def __init__(self, api_key: str | None, timeout_ms: int = 30000, client: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key
        self.timeout_ms = timeout_ms
        self.base_url = "https://api.anthropic.com/v1"
        self.client = client

    @property
    def provider_name(self) -> str:
        return Provider.ANTHROPIC.value

    async def execute(self, model_id: str, request: ProviderRequest) -> ProviderResponse:
        if not self.api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY")

        system_parts = [m["content"] for m in request.messages if m.get("role") == "system"]
        messages = [m for m in request.messages if m.get("role") != "system"]
        if not messages:
            messages = [{"role": "user", "content": request.prompt}]

        payload: dict = {
            "model": model_id,
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "messages": messages,
        }
        if system_parts:
            payload["system"] = "\n".join(system_parts)

        start = time.perf_counter()
        client = self.client or get_http_client()
        response = await client.post(
                f"{self.base_url}/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout_ms / 1000.0,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        if response.status_code == 401:
            raise RuntimeError("anthropic authentication failed")
        if response.status_code == 429:
            raise RuntimeError("anthropic rate limited")
        response.raise_for_status()
        data = response.json()
        content = data.get("content") or []
        text = "".join(block.get("text", "") for block in content if isinstance(block, dict))
        usage = data.get("usage", {}) or {}
        return ProviderResponse(
            text=text,
            usage_input_tokens=int(usage.get("input_tokens", 0)),
            usage_output_tokens=int(usage.get("output_tokens", 0)),
            latency_ms=latency_ms,
            model=model_id,
            provider=Provider.ANTHROPIC,
            provider_request_id=data.get("id"),
        )
