"""Tests for client profiles and passthrough routing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from src.anthropic_compat import AnthropicMessage, MessagesRequest, anthropic_messages_to_prompt
from src.router.clients import ClientConfigError, ClientProfile, load_client_config
from src.router.passthrough import forward_openai_chat, resolve_auth_headers
from src.router.schemas import Tier


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
                    "default": {
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

    monkeypatch.setenv("DISPATCH_PROFILE", "default")
    cfg2 = load_client_config(path)
    assert cfg2.active().is_passthrough
    assert cfg2.active().discover_models is True


def test_env_overrides_models(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DISPATCH_PROFILE", "default")
    monkeypatch.setenv("DISPATCH_MODEL_CHEAP", "my/cheap")
    monkeypatch.setenv("DISPATCH_MODEL_MID", "my/mid")
    monkeypatch.setenv("DISPATCH_MODEL_HARD", "my/hard")
    monkeypatch.setenv("DISPATCH_UPSTREAM_BASE_URL", "https://example.com/v1")
    path = tmp_path / "clients.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "active_profile": "default",
                "profiles": {
                    "default": {
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


def _write_passthrough_clients(path: Path, *, upstream: str = "https://openrouter.ai/api/v1") -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "active_profile": "default",
                "profiles": {
                    "default": {
                        "mode": "passthrough",
                        "protocol": "openai",
                        "upstream_base_url": upstream,
                        "discover_models": True,
                        "models": {"cheap": "", "mid": "", "hard": ""},
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_openai_base_url_is_not_passthrough_upstream(tmp_path: Path, monkeypatch):
    """OPENAI_BASE_URL configures the execute-mode provider, not passthrough upstream.

    Documented upstream knobs are DISPATCH_UPSTREAM_BASE_URL and clients.yaml.
    """
    monkeypatch.setenv("DISPATCH_PROFILE", "default")
    monkeypatch.delenv("DISPATCH_UPSTREAM_BASE_URL", raising=False)
    path = tmp_path / "clients.yaml"
    yaml_upstream = "https://openrouter.ai/api/v1"
    _write_passthrough_clients(path, upstream=yaml_upstream)

    # Absent OPENAI_BASE_URL → YAML upstream.
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert load_client_config(path).active().upstream_base_url == yaml_upstream

    # Present non-loopback OPENAI_BASE_URL (as in .env.example) must not override YAML.
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    assert load_client_config(path).active().upstream_base_url == yaml_upstream

    # Loopback OPENAI_BASE_URL (tools pointing at Dispatch) must not become upstream.
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    assert load_client_config(path).active().upstream_base_url == yaml_upstream

    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    assert load_client_config(path).active().upstream_base_url == yaml_upstream

    # Explicit DISPATCH_UPSTREAM_BASE_URL still wins; loopback Dispatch URL is rejected.
    monkeypatch.setenv("DISPATCH_UPSTREAM_BASE_URL", "https://example.com/v1")
    assert "example.com" in load_client_config(path).active().upstream_base_url

    monkeypatch.setenv("DISPATCH_UPSTREAM_BASE_URL", "http://localhost:8000/v1")
    assert load_client_config(path).active().upstream_base_url == yaml_upstream

    monkeypatch.setenv("DISPATCH_UPSTREAM_BASE_URL", "http://127.0.0.1:8000/v1")
    assert load_client_config(path).active().upstream_base_url == yaml_upstream


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
    assert "anthropic" in cfg.profiles
    assert cfg.profiles["anthropic"].protocol == "anthropic"
    assert cfg.profiles["default"].is_passthrough
    assert cfg.profiles["default"].discover_models is True
    assert cfg.active_profile == "demo"
    assert cfg.active().mode == "execute"


