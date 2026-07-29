from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.router.surface import RequestLimitError, ready_payload, validate_request_limits


def test_validate_request_limits_rejects_blank_prompt():
    svc = SimpleNamespace(config=SimpleNamespace(request_max_prompt_chars=100, request_max_output_tokens=50))
    with pytest.raises(RequestLimitError):
        validate_request_limits(svc, prompt="   ", max_output_tokens=10)


def test_validate_request_limits_rejects_token_overflow():
    svc = SimpleNamespace(config=SimpleNamespace(request_max_prompt_chars=100, request_max_output_tokens=50))
    with pytest.raises(RequestLimitError):
        validate_request_limits(svc, prompt="ok", max_output_tokens=51)


def test_ready_payload_contains_profile_and_models():
    model = SimpleNamespace(
        key="m1",
        provider_model_id="provider/model",
        provider=SimpleNamespace(value="openrouter"),
        tier=SimpleNamespace(value="mid"),
    )
    profile = SimpleNamespace(
        name="default",
        mode="passthrough",
        protocol="openai",
        is_passthrough=True,
        upstream_api_key="",
        upstream_base_url="https://openrouter.ai/api/v1",
        models={"cheap": "a", "mid": "b", "hard": "c"},
        ensure_models=lambda api_key=None: None,
    )
    registry = SimpleNamespace(source="discover", version="v1", enabled_models=lambda: [model])
    svc = SimpleNamespace(active_profile=profile, model_registry=registry)

    payload = ready_payload(svc)
    assert payload["ready"] is True
    assert payload["profile"] == "default"
    assert payload["models"][0]["id"] == "provider/model"
