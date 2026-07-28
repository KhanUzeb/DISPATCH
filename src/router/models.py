from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from .discover import (
    DiscoverConfig,
    DiscoveryCache,
    discover_models,
    load_discover_config,
    pick_tier_representatives,
)
from .schemas import ModelSpec, Provider, Tier


class ModelRegistryError(RuntimeError):
    pass


@dataclass
class ModelRegistry:
    models: list[ModelSpec]
    version: str
    source: str = "static"  # discover | fallback | static
    discover_config: DiscoverConfig | None = None
    fetched_at: float | None = None

    def enabled_models(self) -> list[ModelSpec]:
        return [m for m in self.models if m.enabled]

    def get_by_tier(self, tier: Tier) -> list[ModelSpec]:
        return [m for m in self.enabled_models() if m.tier == tier]

    def get_by_key(self, key: str) -> ModelSpec | None:
        return next((m for m in self.models if m.key == key), None)

    def tier_representatives(self) -> dict[str, str]:
        return pick_tier_representatives(self.enabled_models())

    def refresh(self) -> "ModelRegistry":
        if self.discover_config is None:
            return self
        cache = discover_models(self.discover_config)
        return ModelRegistry(
            models=cache.specs,
            version=cache.version,
            source=cache.source,
            discover_config=self.discover_config,
            fetched_at=cache.fetched_at,
        )


def _specs_from_static_rows(entries: list[dict[str, Any]]) -> list[ModelSpec]:
    seen_keys: set[str] = set()
    models: list[ModelSpec] = []
    for row in entries:
        key = str(row.get("key") or row.get("id") or "").strip()
        if not key:
            raise ModelRegistryError("Model row missing key/id")
        if key in seen_keys:
            raise ModelRegistryError(f"Duplicate model key: {key}")
        seen_keys.add(key)
        in_price = float(row.get("cost_per_1k_input", 0.0))
        out_price = float(row.get("cost_per_1k_output", 0.0))
        if in_price < 0 or out_price < 0:
            raise ModelRegistryError(f"Invalid negative prices for model {key}")
        models.append(
            ModelSpec(
                key=key,
                provider_model_id=str(row.get("provider_model_id", key)),
                provider=Provider(str(row["provider"])),
                tier=Tier(str(row["tier"])),
                enabled=bool(row.get("enabled", True)),
                cost_per_1k_input=in_price,
                cost_per_1k_output=out_price,
                avg_latency_ms=float(row.get("avg_latency_ms", 500)),
                context_window=int(row.get("context_window", 128000)),
                supports_tool_calling=bool(row.get("supports_tool_calling", False)),
                supports_structured_output=bool(row.get("supports_structured_output", False)),
                capabilities=list(row.get("capabilities", [])),
                pricing_validated_at=row.get("pricing_validated_at"),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return models


def _version_for(models: list[ModelSpec]) -> str:
    normalized = yaml.safe_dump(
        sorted(
            [
                {
                    "key": m.key,
                    "provider_model_id": m.provider_model_id,
                    "provider": m.provider.value,
                    "tier": m.tier.value,
                    "enabled": m.enabled,
                    "context_window": m.context_window,
                }
                for m in models
            ],
            key=lambda x: x["key"],
        ),
        sort_keys=True,
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def load_model_registry(path: str | Path, *, discover: bool | None = None) -> ModelRegistry:
    """Load registry from models.yaml — prefers live discovery over static IDs."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ModelRegistryError(f"models.yaml must be a mapping: {path}")

    discover_cfg = parse_or_load_discover(raw)
    if discover is False:
        discover_cfg = replace(discover_cfg, enabled=False)
    elif discover is True:
        discover_cfg = replace(discover_cfg, enabled=True)

    # New-style discovery config (source: discover) or legacy models: list only.
    legacy_models = raw.get("models") if isinstance(raw.get("models"), list) else None
    fallback = raw.get("fallback_models") if isinstance(raw.get("fallback_models"), list) else None

    if discover_cfg.enabled:
        try:
            cache: DiscoveryCache = discover_models(discover_cfg)
            return ModelRegistry(
                models=cache.specs,
                version=cache.version,
                source=cache.source,
                discover_config=discover_cfg,
                fetched_at=cache.fetched_at,
            )
        except Exception as exc:
            # Fall through to static lists.
            if not (fallback or legacy_models):
                raise ModelRegistryError(str(exc)) from exc

    entries = list(fallback or legacy_models or [])
    if not entries:
        raise ModelRegistryError(
            "configs/models.yaml needs discover.providers or fallback_models/models"
        )
    models = _specs_from_static_rows(entries)
    return ModelRegistry(
        models=models,
        version=_version_for(models),
        source="static",
        discover_config=discover_cfg,
        fetched_at=None,
    )


def parse_or_load_discover(raw: dict[str, Any]) -> DiscoverConfig:
    from .discover import parse_discover_config

    return parse_discover_config(raw)


# Re-export for callers that imported load_discover_config via models before.
__all__ = [
    "ModelRegistry",
    "ModelRegistryError",
    "load_model_registry",
    "load_discover_config",
]
