from __future__ import annotations

import time

import httpx

from ..schemas import Provider, ProviderRequest, ProviderResponse
from .base import ProviderAdapter
from ..http_client import get_http_client


class GoogleAdapter(ProviderAdapter):
    """Google Gemini generateContent API adapter."""

    def __init__(self, api_key: str | None, timeout_ms: int = 30000, client: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key
        self.timeout_ms = timeout_ms
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.client = client

    @property
    def provider_name(self) -> str:
        return Provider.GOOGLE.value

    async def execute(self, model_id: str, request: ProviderRequest) -> ProviderResponse:
        if not self.api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY")

        contents = []
        for message in request.messages:
            role = "user" if message.get("role") != "assistant" else "model"
            contents.append({"role": role, "parts": [{"text": message.get("content", "")}]})
        if not contents:
            contents = [{"role": "user", "parts": [{"text": request.prompt}]}]

        url = f"{self.base_url}/models/{model_id}:generateContent"
        start = time.perf_counter()
        client = self.client or get_http_client()
        response = await client.post(
                url,
                params={"key": self.api_key},
                headers={"Content-Type": "application/json"},
                json={
                    "contents": contents,
                    "generationConfig": {
                        "temperature": request.temperature,
                        "maxOutputTokens": request.max_output_tokens,
                    },
                },
                timeout=self.timeout_ms / 1000.0,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        if response.status_code == 401:
            raise RuntimeError("google authentication failed")
        if response.status_code == 429:
            raise RuntimeError("google rate limited")
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates") or []
        parts = (((candidates[0] or {}).get("content") or {}).get("parts") or []) if candidates else []
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        usage = data.get("usageMetadata", {}) or {}
        return ProviderResponse(
            text=text,
            usage_input_tokens=int(usage.get("promptTokenCount", 0)),
            usage_output_tokens=int(usage.get("candidatesTokenCount", 0)),
            latency_ms=latency_ms,
            model=model_id,
            provider=Provider.GOOGLE,
            provider_request_id=None,
        )
