"""Tests for client profiles and passthrough routing."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.router.clients import ClientConfigError, ClientProfile, load_client_config
from src.router.passthrough import resolve_auth_headers
from src.router.schemas import Tier
from src.anthropic_compat import MessagesRequest, AnthropicMessage, anthropic_messages_to_prompt


def test_load_clients_yaml(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("DISPATCH_PROFILE", raising=False)
    monkeypatch.delenv("DISPATCH_MODE", raising=False)
    path = tmp_path / "clients.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "active_profile": "demo",
                "profiles": {
                    "demo": {
                        "mode": "execute",
                        "protocol": "openai",
                        "upstream_base_url": "",
                        "discover_models": True,
                        "models": {"cheap": "", "mid": "", "hard": ""},
                    },
                    "opencode": {
                        "mode": "passthrough",
                        "protocol": "openai",
                        "upstream_base_url": "https://openrouter.ai/api/v1",
                        "discover_models": True,
                        "models": {"cheap": "", "mid": "", "hard": ""},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_client_config(path)
    assert cfg.active_profile == "demo"
    assert cfg.active().mode == "execute"

    monkeypatch.setenv("DISPATCH_PROFILE", "opencode")
    cfg2 = load_client_config(path)
    assert cfg2.active().is_passthrough
    assert cfg2.active().discover_models is True


def test_env_overrides_models(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DISPATCH_PROFILE", "opencode")
    monkeypatch.setenv("DISPATCH_MODEL_CHEAP", "my/cheap")
    monkeypatch.setenv("DISPATCH_MODEL_MID", "my/mid")
    monkeypatch.setenv("DISPATCH_MODEL_HARD", "my/hard")
    monkeypatch.setenv("DISPATCH_UPSTREAM_BASE_URL", "https://example.com/v1")
    path = tmp_path / "clients.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "active_profile": "opencode",
                "profiles": {
                    "opencode": {
                        "mode": "passthrough",
                        "protocol": "openai",
                        "upstream_base_url": "https://openrouter.ai/api/v1",
                        "discover_models": True,
                        "models": {"cheap": "", "mid": "", "hard": ""},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    profile = load_client_config(path).active()
    assert profile.models["cheap"] == "my/cheap"
    assert profile.has_all_tier_models()
    assert "example.com" in profile.upstream_base_url


def test_passthrough_requires_models_or_discover(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("DISPATCH_PROFILE", raising=False)
    monkeypatch.delenv("DISPATCH_DISCOVER_MODELS", raising=False)
    path = tmp_path / "clients.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "active_profile": "bad",
                "profiles": {
                    "bad": {
                        "mode": "passthrough",
                        "protocol": "openai",
                        "upstream_base_url": "https://api.openai.com/v1",
                        "discover_models": False,
                        "models": {"cheap": "x", "mid": "", "hard": "z"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ClientConfigError):
        load_client_config(path)


def test_resolve_auth_forwards_bearer():
    profile = ClientProfile(
        name="t",
        mode="passthrough",
        protocol="openai",
        upstream_base_url="https://api.openai.com/v1",
        upstream_api_key="",
        models={"cheap": "a", "mid": "b", "hard": "c"},
    )
    headers = resolve_auth_headers(
        profile,
        incoming_authorization="Bearer sk-live",
        incoming_api_key=None,
        incoming_auth_token=None,
        protocol="openai",
    )
    assert headers["Authorization"] == "Bearer sk-live"


def test_resolve_auth_prefers_static_key():
    profile = ClientProfile(
        name="t",
        mode="passthrough",
        protocol="openai",
        upstream_base_url="https://api.openai.com/v1",
        upstream_api_key="sk-static",
        models={"cheap": "a", "mid": "b", "hard": "c"},
    )
    headers = resolve_auth_headers(
        profile,
        incoming_authorization="Bearer sk-live",
        incoming_api_key=None,
        incoming_auth_token=None,
        protocol="openai",
    )
    assert headers["Authorization"] == "Bearer sk-static"


def test_anthropic_prompt_includes_system():
    req = MessagesRequest(
        model="dispatch",
        system="You are helpful",
        messages=[
            AnthropicMessage(role="user", content="earlier"),
            AnthropicMessage(role="assistant", content="response"),
            AnthropicMessage(role="user", content="hi"),
        ],
    )
    prompt = anthropic_messages_to_prompt(req)
    assert "system: You are helpful" in prompt
    assert "user: hi" in prompt
    assert "earlier" not in prompt
    assert "response" not in prompt


def test_project_clients_yaml_loads(monkeypatch):
    monkeypatch.delenv("DISPATCH_PROFILE", raising=False)
    monkeypatch.delenv("DISPATCH_MODE", raising=False)
    monkeypatch.delenv("DISPATCH_MODEL_CHEAP", raising=False)
    monkeypatch.delenv("DISPATCH_MODEL_MID", raising=False)
    monkeypatch.delenv("DISPATCH_MODEL_HARD", raising=False)
    root = Path(__file__).resolve().parents[1]
    cfg = load_client_config(root / "configs" / "clients.yaml")
    assert "demo" in cfg.profiles
    assert "default" in cfg.profiles
    assert "claude-code" in cfg.profiles
    assert cfg.profiles["claude-code"].protocol == "anthropic"
    assert cfg.profiles["default"].is_passthrough
    assert cfg.profiles["default"].discover_models is True
    assert cfg.active_profile == "demo"
    assert cfg.active().mode == "execute"
