"""Early Content-Length guard must reject oversized bodies before classification."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from src.api import app
from src.router.security import SlidingWindowRateLimiter, require_router_auth


@pytest.mark.asyncio
async def test_oversized_actual_body_rejected_before_classifier(monkeypatch):
    monkeypatch.setenv("ROUTER_MAX_REQUEST_BYTES", "200")
    monkeypatch.setattr("src.api.rate_limiter", SlidingWindowRateLimiter(limit_per_minute=0))

    classify_calls = 0

    class FakeService:
        active_profile = None
        config = SimpleNamespace(request_max_prompt_chars=100_000, request_max_output_tokens=4096)

        def decide(self, *args, **kwargs):
            nonlocal classify_calls
            classify_calls += 1
            raise AssertionError("classifier must not run for oversized bodies")

    monkeypatch.setattr("src.api._service_or_raise", lambda: FakeService())
    app.dependency_overrides[require_router_auth] = lambda: None

    payload = {
        "model": "dispatch",
        "messages": [{"role": "user", "content": "x" * 500}],
    }
    assert len(json.dumps(payload)) > 200

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/v1/messages", json=payload)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 413, response.text
    assert response.json()["error_category"] == "payload_too_large"
    assert classify_calls == 0
