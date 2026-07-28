from __future__ import annotations

import time

import requests

from ..schemas import Provider, ProviderRequest, ProviderResponse
from .base import ProviderAdapter


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
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_ms = timeout_ms
        self.extra_headers = extra_headers or {}
        self.require_api_key = require_api_key

    @property
    def provider_name(self) -> str:
        return self.provider.value

    def execute(self, model_id: str, request: ProviderRequest) -> ProviderResponse:
        if self.require_api_key and not self.api_key:
            raise RuntimeError(f"Missing API key for provider {self.provider.value}")

        headers = {
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        start = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/chat/completions",
            timeout=self.timeout_ms / 1000.0,
            headers=headers,
            json={
                "model": model_id,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_output_tokens,
            },
        )
        latency_ms = (time.perf_counter() - start) * 1000
        if response.status_code == 401:
            raise RuntimeError(f"{self.provider.value} authentication failed")
        if response.status_code == 429:
            raise RuntimeError(f"{self.provider.value} rate limited")
        response.raise_for_status()
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
