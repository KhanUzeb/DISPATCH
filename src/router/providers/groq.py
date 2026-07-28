from __future__ import annotations

import time

from .base import ProviderAdapter
from ..schemas import Provider, ProviderRequest, ProviderResponse


class GroqAdapter(ProviderAdapter):
    def __init__(self, api_key: str | None, timeout_ms: int = 30000) -> None:
        self.api_key = api_key
        self.timeout_ms = timeout_ms
        self._client = None

    @property
    def provider_name(self) -> str:
        return Provider.GROQ.value

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError("Missing GROQ_API_KEY")
        from groq import Groq

        self._client = Groq(api_key=self.api_key, timeout=self.timeout_ms / 1000.0)
        return self._client

    def execute(self, model_id: str, request: ProviderRequest) -> ProviderResponse:
        start = time.perf_counter()
        client = self._get_client()
        resp = client.chat.completions.create(
            model=model_id,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_output_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        usage = resp.usage
        return ProviderResponse(
            text=resp.choices[0].message.content or "",
            usage_input_tokens=usage.prompt_tokens if usage else 0,
            usage_output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            model=model_id,
            provider=Provider.GROQ,
            provider_request_id=getattr(resp, "id", None),
        )
