from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.api import complete
from src.mcp_server import _tool_error, complete as mcp_complete
from src.openai_compat import ChatMessage, messages_to_prompt
from src.router.clients import ClientProfile


class _FakeCompleteRequest:
    prompt = "hello"
    max_output_tokens = 64
    max_cost_usd = None
    max_latency_ms = None
    min_context_window = None
    expects_structured_output = False
    require_tool_calling = False
    include_diagnostics = False


def test_api_complete_rejects_passthrough_profiles(monkeypatch):
    profile = ClientProfile(
        name="opencode",
        mode="passthrough",
        protocol="openai",
        upstream_base_url="https://openrouter.ai/api/v1",
        upstream_api_key="",
        models={"cheap": "a", "mid": "b", "hard": "c"},
    )
    svc = SimpleNamespace(
        active_profile=profile,
        config=SimpleNamespace(request_max_prompt_chars=10000, request_max_output_tokens=4096),
    )
    monkeypatch.setattr("src.api._service_or_raise", lambda: svc)

    with pytest.raises(HTTPException) as exc:
        complete(_FakeCompleteRequest(), None)  # type: ignore[arg-type]
    assert exc.value.status_code == 400
    detail = exc.value.detail
    assert detail["error_category"] == "invalid_request"
    assert "/v1/chat/completions" in detail["message"]


def test_mcp_complete_rejects_passthrough_profiles(monkeypatch):
    profile = ClientProfile(
        name="claude-code",
        mode="passthrough",
        protocol="anthropic",
        upstream_base_url="https://api.anthropic.com",
        upstream_api_key="",
        models={"cheap": "a", "mid": "b", "hard": "c"},
    )
    svc = SimpleNamespace(
        active_profile=profile,
        config=SimpleNamespace(request_max_prompt_chars=10000, request_max_output_tokens=4096),
    )
    monkeypatch.setattr("src.mcp_server._service_or_error", lambda: svc)
    result = mcp_complete(prompt="hello", max_output_tokens=32)
    assert "error" in result
    assert result["error"]["category"] == "invalid_request"
    assert "/v1/messages" in result["error"]["message"]


def test_mcp_tool_error_shape():
    err = _tool_error("boom", category="service_unavailable")
    assert err == {
        "error": {"category": "service_unavailable", "message": "boom", "retryable": False}
    }


def test_messages_to_prompt_ignores_empty_roles():
    prompt = messages_to_prompt(
        [
            ChatMessage(role="system", content=""),
            ChatMessage(role="assistant", content="ignored"),
            ChatMessage(role="user", content="do the thing"),
        ]
    )
    assert prompt == "do the thing"
