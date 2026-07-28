"""Client profiles for universal router / passthrough mode.

Coding tools (OpenCode, Codex, Claude Code, Cursor) point at Dispatch as a
router. Dispatch classifies the prompt into cheap/mid/hard, maps that tier to
a model discovered from the tool's upstream (or DISPATCH_MODEL_* overrides),
and forwards the request.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .discover import TieringConfig, discover_for_upstream
from .schemas import Provider, Tier

logger = logging.getLogger(__name__)


class ClientConfigError(RuntimeError):
    pass


@dataclass
class ClientProfile:
    name: str
    mode: str  # passthrough | execute
    protocol: str  # openai | anthropic
    upstream_base_url: str
    upstream_api_key: str
    models: dict[str, str]  # tier -> model id (may be filled by discovery)
    fallbacks: dict[str, list[str]] = field(default_factory=dict)
    discover_models: bool = True
    free_only: bool = False
    _discovered: bool = False

    @property
    def is_passthrough(self) -> bool:
        return self.mode == "passthrough"

    def model_for_tier(self, tier: Tier) -> str | None:
        model = (self.models.get(tier.value) or "").strip()
        return model or None

    def fallback_models(self, tier: Tier) -> list[str]:
        return [m for m in self.fallbacks.get(tier.value, []) if str(m).strip()]

    def has_all_tier_models(self) -> bool:
        return all((self.models.get(t) or "").strip() for t in ("cheap", "mid", "hard"))

    def ensure_models(self, *, api_key: str | None = None) -> None:
        """Fetch tier models from the tool upstream when not already set."""
        if self.has_all_tier_models():
            return
        if not self.discover_models:
            raise ClientConfigError(
                f"Profile '{self.name}' is missing models.cheap/mid/hard and discover_models is false"
            )
        if self.mode == "execute":
            # Execute mode uses ModelRegistry discovery instead.
            return
        if not self.upstream_base_url and self.protocol != "anthropic":
            raise ClientConfigError(
                f"Profile '{self.name}' needs upstream_base_url to discover models"
            )
        key = (api_key or self.upstream_api_key or "").strip()
        provider = Provider.OPENROUTER if "openrouter.ai" in self.upstream_base_url else Provider.OPENAI_COMPATIBLE
        if self.protocol == "anthropic":
            provider = Provider.ANTHROPIC
        try:
            picked = discover_for_upstream(
                base_url=self.upstream_base_url or "https://api.anthropic.com",
                api_key=key,
                protocol=self.protocol,
                free_only=self.free_only,
                tiering=TieringConfig(prefer_free=self.free_only),
                provider=provider,
            )
        except Exception as exc:
            raise ClientConfigError(
                f"Failed to discover models for profile '{self.name}' from "
                f"{self.upstream_base_url}: {exc}"
            ) from exc
        for tier in ("cheap", "mid", "hard"):
            if not (self.models.get(tier) or "").strip() and picked.get(tier):
                self.models[tier] = picked[tier]
        if not self.has_all_tier_models():
            raise ClientConfigError(
                f"Upstream for '{self.name}' did not yield models for all tiers. "
                f"Got: {self.models}. Set DISPATCH_MODEL_CHEAP/MID/HARD."
            )
        self._discovered = True
        logger.info(
            "Discovered tier models for %s: cheap=%s mid=%s hard=%s",
            self.name,
            self.models.get("cheap"),
            self.models.get("mid"),
            self.models.get("hard"),
        )


@dataclass(frozen=True)
class ClientConfig:
    active_profile: str
    profiles: dict[str, ClientProfile]
    path: Path
    version: str

    def active(self) -> ClientProfile:
        if self.active_profile not in self.profiles:
            raise ClientConfigError(
                f"Unknown active profile '{self.active_profile}'. "
                f"Known: {sorted(self.profiles)}"
            )
        return self.profiles[self.active_profile]


def _env_expand(value: str) -> str:
    if not value:
        return ""
    if value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    return value


def _normalize_base_url(url: str, *, protocol: str) -> str:
    url = (url or "").strip().rstrip("/")
    if not url:
        return ""
    if protocol == "openai" and url.endswith("/v1/"):
        return url.rstrip("/")
    return url


def _parse_profile(name: str, raw: dict[str, Any]) -> ClientProfile:
    mode = str(raw.get("mode", "passthrough")).strip().lower()
    if mode not in {"passthrough", "execute"}:
        raise ClientConfigError(f"Profile '{name}': mode must be passthrough|execute")
    protocol = str(raw.get("protocol", "openai")).strip().lower()
    if protocol not in {"openai", "anthropic"}:
        raise ClientConfigError(f"Profile '{name}': protocol must be openai|anthropic")

    models_raw = raw.get("models") or {}
    models = {
        "cheap": str(models_raw.get("cheap", "") or "").strip(),
        "mid": str(models_raw.get("mid", "") or "").strip(),
        "hard": str(models_raw.get("hard", "") or "").strip(),
    }
    fallbacks_raw = raw.get("fallbacks") or {}
    fallbacks = {
        "cheap": [str(x).strip() for x in (fallbacks_raw.get("cheap") or []) if str(x).strip()],
        "mid": [str(x).strip() for x in (fallbacks_raw.get("mid") or []) if str(x).strip()],
        "hard": [str(x).strip() for x in (fallbacks_raw.get("hard") or []) if str(x).strip()],
    }
    discover_models = bool(raw.get("discover_models", True))
    free_only = bool(raw.get("free_only", False))

    if mode == "passthrough" and not all(models.values()) and not discover_models:
        raise ClientConfigError(
            f"Profile '{name}': passthrough needs models.cheap/mid/hard or discover_models: true"
        )

    return ClientProfile(
        name=name,
        mode=mode,
        protocol=protocol,
        upstream_base_url=_normalize_base_url(
            _env_expand(str(raw.get("upstream_base_url", "") or "")),
            protocol=protocol,
        ),
        upstream_api_key=_env_expand(str(raw.get("upstream_api_key", "") or "")),
        models=models,
        fallbacks=fallbacks,
        discover_models=discover_models,
        free_only=free_only,
    )


def _apply_env_overrides(profile: ClientProfile) -> ClientProfile:
    """Env wins over YAML so installers can configure without editing files."""
    mode = (os.environ.get("DISPATCH_MODE") or profile.mode).strip().lower()
    if mode not in {"passthrough", "execute"}:
        mode = profile.mode

    upstream = (
        os.environ.get("DISPATCH_UPSTREAM_BASE_URL")
        or profile.upstream_base_url
    )
    # Ignore loopback OPENAI_BASE_URL — that points at Dispatch itself.
    env_openai = os.environ.get("OPENAI_BASE_URL", "")
    if env_openai and "localhost:8000" not in env_openai and "127.0.0.1:8000" not in env_openai:
        if not os.environ.get("DISPATCH_UPSTREAM_BASE_URL"):
            # Only use OPENAI_BASE_URL as upstream when explicitly not Dispatch.
            pass

    if upstream and ("localhost:8000" in upstream or "127.0.0.1:8000" in upstream):
        upstream = profile.upstream_base_url

    api_key = os.environ.get("DISPATCH_UPSTREAM_API_KEY") or profile.upstream_api_key

    models = dict(profile.models)
    for tier, env_key in (
        ("cheap", "DISPATCH_MODEL_CHEAP"),
        ("mid", "DISPATCH_MODEL_MID"),
        ("hard", "DISPATCH_MODEL_HARD"),
    ):
        override = (os.environ.get(env_key) or "").strip()
        if override:
            models[tier] = override

    protocol = (os.environ.get("DISPATCH_PROTOCOL") or profile.protocol).strip().lower()
    if protocol not in {"openai", "anthropic"}:
        protocol = profile.protocol

    discover_models = profile.discover_models
    if os.environ.get("DISPATCH_DISCOVER_MODELS", "").lower() in {"0", "false", "no"}:
        discover_models = False
    if os.environ.get("DISPATCH_DISCOVER_MODELS", "").lower() in {"1", "true", "yes"}:
        discover_models = True

    if mode == "passthrough" and not all(models.values()) and not discover_models:
        if profile.mode == "execute":
            mode = "execute"
        else:
            raise ClientConfigError(
                "Passthrough needs DISPATCH_MODEL_CHEAP/MID/HARD or discover_models"
            )

    return ClientProfile(
        name=profile.name,
        mode=mode,
        protocol=protocol,
        upstream_base_url=_normalize_base_url(str(upstream or ""), protocol=protocol),
        upstream_api_key=str(api_key or ""),
        models=models,
        fallbacks=profile.fallbacks,
        discover_models=discover_models,
        free_only=profile.free_only,
    )


def load_client_config(path: str | Path) -> ClientConfig:
    path = Path(path)
    if not path.exists():
        demo = ClientProfile(
            name="demo",
            mode="execute",
            protocol="openai",
            upstream_base_url="",
            upstream_api_key="",
            models={"cheap": "", "mid": "", "hard": ""},
            fallbacks={"cheap": [], "mid": [], "hard": []},
            discover_models=True,
            free_only=True,
        )
        return ClientConfig(
            active_profile="demo",
            profiles={"demo": demo},
            path=path,
            version="missing",
        )

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ClientConfigError(f"clients.yaml must be a mapping: {path}")

    profiles_raw = raw.get("profiles") or {}
    if not isinstance(profiles_raw, dict) or not profiles_raw:
        raise ClientConfigError("clients.yaml must contain a non-empty profiles map")

    profiles = {name: _parse_profile(name, row or {}) for name, row in profiles_raw.items()}
    active = (os.environ.get("DISPATCH_PROFILE") or raw.get("active_profile") or "default").strip()
    if active not in profiles:
        raise ClientConfigError(
            f"active_profile '{active}' not found. Known: {sorted(profiles)}"
        )

    profiles = dict(profiles)
    profiles[active] = _apply_env_overrides(profiles[active])

    version = yaml.safe_dump(
        {
            "active": active,
            "profile": {
                "name": profiles[active].name,
                "mode": profiles[active].mode,
                "models": profiles[active].models,
                "discover": profiles[active].discover_models,
            },
        },
        sort_keys=True,
    )
    import hashlib

    return ClientConfig(
        active_profile=active,
        profiles=profiles,
        path=path,
        version=hashlib.sha256(version.encode("utf-8")).hexdigest()[:16],
    )
