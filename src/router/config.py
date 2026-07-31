from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .schemas import Provider


class ConfigError(RuntimeError):
    pass


KNOWN_PROVIDERS = [p.value for p in Provider]
DEFAULT_PROVIDER_TIMEOUT_MS = 30000
DEFAULT_PROVIDER_RETRIES = 0
DEFAULT_MAX_REQUEST_BYTES = 2_000_000


def max_request_bytes() -> int:
    """Ceiling for raw HTTP request bodies (ROUTER_MAX_REQUEST_BYTES)."""
    raw = os.environ.get("ROUTER_MAX_REQUEST_BYTES", "").strip()
    if not raw:
        return DEFAULT_MAX_REQUEST_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_REQUEST_BYTES
    return value if value > 0 else DEFAULT_MAX_REQUEST_BYTES


@dataclass(frozen=True)
class RouterRuntimeConfig:
    project_root: Path
    routes_path: Path
    models_path: Path
    clients_path: Path
    encoder_model: str
    encoder_device: str | None
    default_threshold: float
    top_k: int
    route_aggregation: str
    policy_version: str
    policy_diversify: bool
    max_fallbacks: int
    cache_mode: str
    decision_cache_ttl_seconds: int
    telemetry_mode: str
    request_max_prompt_chars: int
    request_max_output_tokens: int
    request_max_bytes: int
    provider_timeouts_ms: dict[str, int]
    provider_retries: dict[str, int]
    config_version: str


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file must contain a mapping: {path}")
    return data


def _hash_obj(obj: Any) -> str:
    dumped = yaml.safe_dump(obj, sort_keys=True)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:16]


def _provider_map(section: dict[str, Any], default: int) -> dict[str, int]:
    values: dict[str, int] = {name: default for name in KNOWN_PROVIDERS}
    for name, raw in (section or {}).items():
        values[str(name)] = int(raw)
    return values


def load_config(project_root: str | Path | None = None) -> RouterRuntimeConfig:
    load_dotenv()
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    router_yaml = root / "configs" / "router.yaml"
    config = _read_yaml(router_yaml)

    encoder = config.get("encoder", {})
    thresholds = config.get("thresholds", {})
    policy = config.get("policy", {})
    cache = config.get("cache", {})
    telemetry = config.get("telemetry", {})
    request_limits = config.get("request_limits", {})
    provider = config.get("provider", {})

    routes_path = (root / config.get("routes_path", "configs/routes.yaml")).resolve()
    models_path = (root / config.get("models_path", "configs/models.yaml")).resolve()
    clients_path = (root / config.get("clients_path", "configs/clients.yaml")).resolve()
    if not routes_path.exists():
        raise ConfigError(f"Routes file does not exist: {routes_path}")
    if not models_path.exists():
        raise ConfigError(f"Models file does not exist: {models_path}")

    default_threshold = float(thresholds.get("default", 0.7))
    if not 0.0 <= default_threshold <= 1.0:
        raise ConfigError("thresholds.default must be in [0, 1]")

    top_k = int(thresholds.get("top_k", 5))
    if top_k <= 0:
        raise ConfigError("thresholds.top_k must be > 0")

    decision_cache_ttl_seconds = int(cache.get("decision_ttl_seconds", 300))
    if decision_cache_ttl_seconds <= 0:
        raise ConfigError("cache.decision_ttl_seconds must be > 0")

    provider_timeouts_ms = _provider_map(provider.get("timeouts_ms", {}), DEFAULT_PROVIDER_TIMEOUT_MS)
    provider_retries = _provider_map(provider.get("retries", {}), DEFAULT_PROVIDER_RETRIES)

    if any(v <= 0 for v in provider_timeouts_ms.values()):
        raise ConfigError("provider timeouts must be > 0")
    if any(v < 0 for v in provider_retries.values()):
        raise ConfigError("provider retries must be >= 0")

    env_keys = [
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "TOGETHER_API_KEY",
        "DEEPSEEK_API_KEY",
        "FIREWORKS_API_KEY",
        "MISTRAL_API_KEY",
        "OPENAI_COMPATIBLE_API_KEY",
    ]
    merged_for_version = {
        "router": config,
        "env_presence": {key: bool(os.environ.get(key)) for key in env_keys},
    }

    return RouterRuntimeConfig(
        project_root=root,
        routes_path=routes_path,
        models_path=models_path,
        clients_path=clients_path,
        encoder_model=str(encoder.get("model", "sentence-transformers/all-MiniLM-L6-v2")),
        encoder_device=encoder.get("device"),
        default_threshold=default_threshold,
        top_k=top_k,
        route_aggregation=str(thresholds.get("aggregation", "max")),
        policy_version=str(policy.get("version", "v2")),
        policy_diversify=bool(policy.get("diversify", True)),
        max_fallbacks=int(policy.get("max_fallbacks", 3)),
        cache_mode=str(cache.get("mode", "memory")),
        decision_cache_ttl_seconds=decision_cache_ttl_seconds,
        telemetry_mode=str(telemetry.get("mode", "noop")),
        request_max_prompt_chars=int(request_limits.get("max_prompt_chars", 200000)),
        request_max_output_tokens=int(request_limits.get("max_output_tokens", 131072)),
        request_max_bytes=max_request_bytes(),
        provider_timeouts_ms=provider_timeouts_ms,
        provider_retries=provider_retries,
        config_version=_hash_obj(merged_for_version),
    )
