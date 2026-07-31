"""RouterRuntimeConfig loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.router.config import (
    DEFAULT_MAX_REQUEST_BYTES,
    DEFAULT_PROVIDER_RETRIES,
    DEFAULT_PROVIDER_TIMEOUT_MS,
    KNOWN_PROVIDERS,
    ConfigError,
    _provider_map,
    load_config,
    max_request_bytes,
)


def _write_router_tree(
    root: Path,
    *,
    router: dict | None = None,
    routes: dict | None = None,
    models: dict | None = None,
) -> None:
    configs = root / "configs"
    configs.mkdir(parents=True)
    (configs / "routes.yaml").write_text(
        yaml.safe_dump(routes if routes is not None else {"routes": []}),
        encoding="utf-8",
    )
    (configs / "models.yaml").write_text(
        yaml.safe_dump(
            models
            if models is not None
            else {"source": "static", "fallback_models": [{"key": "m", "provider": "groq", "tier": "cheap"}]}
        ),
        encoding="utf-8",
    )
    (configs / "clients.yaml").write_text(
        yaml.safe_dump({"active_profile": "demo", "profiles": {"demo": {"mode": "execute"}}}),
        encoding="utf-8",
    )
    default_router = {
        "routes_path": "configs/routes.yaml",
        "models_path": "configs/models.yaml",
        "clients_path": "configs/clients.yaml",
        "thresholds": {"default": 0.5, "top_k": 3},
        "cache": {"decision_ttl_seconds": 60},
        "provider": {"timeouts_ms": {"groq": 1000}, "retries": {"groq": 1}},
    }
    (configs / "router.yaml").write_text(
        yaml.safe_dump(router if router is not None else default_router),
        encoding="utf-8",
    )


def test_load_config_happy_path(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ROUTER_MAX_REQUEST_BYTES", raising=False)
    _write_router_tree(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.project_root == tmp_path.resolve()
    assert cfg.default_threshold == 0.5
    assert cfg.top_k == 3
    assert cfg.decision_cache_ttl_seconds == 60
    assert cfg.provider_timeouts_ms["groq"] == 1000
    assert cfg.provider_timeouts_ms["openai"] == DEFAULT_PROVIDER_TIMEOUT_MS
    assert cfg.provider_retries["groq"] == 1
    assert cfg.provider_retries["openai"] == DEFAULT_PROVIDER_RETRIES
    assert cfg.config_version
    assert cfg.routes_path.exists()
    assert cfg.models_path.exists()


def test_load_config_missing_router_yaml(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path)


@pytest.mark.parametrize(
    "thresholds,match",
    [
        ({"default": 1.5, "top_k": 3}, r"thresholds\.default"),
        ({"default": -0.1, "top_k": 3}, r"thresholds\.default"),
        ({"default": 0.5, "top_k": 0}, r"thresholds\.top_k"),
    ],
)
def test_load_config_threshold_validation(tmp_path: Path, thresholds: dict, match: str):
    _write_router_tree(tmp_path, router={"thresholds": thresholds, "cache": {"decision_ttl_seconds": 10}})
    with pytest.raises(ConfigError, match=match):
        load_config(tmp_path)


