from __future__ import annotations

import time

import requests

from .base import ProviderAdapter
from ..schemas import Provider, ProviderRequest, ProviderResponse


class OpenRouterAdapter(ProviderAdapter):
    def __init__(self, api_key: str | None, timeout_ms: int = 30000) -> None:
        self.api_key = api_key
        self.timeout_ms = timeout_ms

    @property
    def provider_name(self) -> str:
        return Provider.OPENROUTER.value

    def execute(self, model_id: str, request: ProviderRequest) -> ProviderResponse:
        if not self.api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY")

        start = time.perf_counter()
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            timeout=self.timeout_ms / 1000.0,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_id,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_output_tokens,
            },
        )
        latency_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {})
        return ProviderResponse(
            text=(data.get("choices", [{}])[0].get("message", {}).get("content") or ""),
            usage_input_tokens=int(usage.get("prompt_tokens", 0)),
            usage_output_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency_ms,
            model=model_id,
            provider=Provider.OPENROUTER,
            provider_request_id=data.get("id"),
        )
