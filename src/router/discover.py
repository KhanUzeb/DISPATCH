"""Live model discovery from providers and coding-tool upstreams.

models.yaml no longer hardcodes model IDs. At startup (and on refresh) we
fetch `/v1/models` (or OpenRouter's richer catalog) and bucket into
cheap / mid / hard using size + name heuristics.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import yaml

from .schemas import ModelSpec, Provider, Tier

logger = logging.getLogger(__name__)

_PARAM_RE = re.compile(r"(?i)(?:^|[^a-z0-9])(\d+(?:\.\d+)?)\s*b(?:illion)?(?:[^a-z0-9]|$)")
_CHEAP_HINTS = (
    "mini",
    "nano",
    "tiny",
    "small",
    "haiku",
    "flash",
    "instant",
    "lite",
    "fast",
    "8b",
    "7b",
    "3b",
    "1b",
    "9b",
)
_HARD_HINTS = (
    "opus",
    "o1",
    "o3",
    "o4",
    "120b",
    "405b",
    "reasoning",
    "r1",
    "deepseek-r",
    "codex",
    "pro-",
    "-pro",
    # Reasoning / CoT models — keep out of cheap/mid demo seats.
    "gpt-oss",
    "qwen3",
)
_MID_HINTS = ("sonnet", "70b", "72b", "80b", "versatile", "gpt-4o", "gpt-4.1", "32b", "34b")

# Models that dump chain-of-thought / <think> blocks — bad for snappy demos.
_REASONING_ID_SUBSTR = (
    "gpt-oss",
    "qwen3",
    "o1-",
    "o3-",
    "o4-",
    "deepseek-r",
    "reasoning",
)

# Non-chat / non-text models that must never enter the demo registry.
_BLOCKED_ID_SUBSTR = (
    "lyria",
    "whisper",
    "tts",
    "embedding",
    "embed",
    "rerank",
    "moderation",
    "clip",
    "flux",
    "sdxl",
    "stable-diffusion",
    "dall-e",
    "imagen",
    "video",
    "audio",
    "music",
    "asr",
    "transcri",
    "vision-only",
    "image-edit",
    "openrouter/free",  # unstable auto-router; picks random free endpoints
    "guard",
    "prompt-guard",
    # Chronically slow / flaky OpenRouter free endpoints that ruin demos.
    "poolside",
    "laguna",
    "nemotron",
    # Groq agent/compound systems — not snappy chat completions for demos.
    "groq/compound",
    "compound-mini",
)

_CHAT_OUTPUT_HINTS = ("text", "chat", "completion")
_NON_CHAT_MODALITIES = ("audio", "image", "video", "music", "embedding", "speech")


@dataclass(frozen=True)
class DiscoveredModel:
    model_id: str
    provider: Provider
    context_window: int = 128000
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    supports_tool_calling: bool = True
    supports_structured_output: bool = True
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DiscoverSource:
    name: str
    provider: Provider
    base_url: str
    api_key_env: str
    free_only: bool = True
    prefer_suffix: str = ""
    enabled: bool = True


@dataclass(frozen=True)
class TieringConfig:
    strategy: str = "size_then_hints"
    cheap_max_params_b: float = 20.0
    mid_max_params_b: float = 80.0
    max_per_tier: int = 8
    prefer_free: bool = True


@dataclass
class DiscoverConfig:
    enabled: bool = True
    cache_ttl_seconds: int = 3600
    sources: list[DiscoverSource] = field(default_factory=list)
    tiering: TieringConfig = field(default_factory=TieringConfig)
    fallback_models: list[dict[str, Any]] = field(default_factory=list)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def estimate_params_b(model_id: str, name: str = "") -> float | None:
    text = f"{model_id} {name}"
    matches = _PARAM_RE.findall(text.replace("_", "-"))
    if not matches:
        return None
    try:
        return max(float(m) for m in matches)
    except ValueError:
        return None


def assign_tier(model: DiscoveredModel, tiering: TieringConfig) -> Tier:
    blob = f"{model.model_id} {model.name}".lower()
    if any(h in blob for h in _HARD_HINTS):
        return Tier.HARD
    if any(h in blob for h in _CHEAP_HINTS) and not any(h in blob for h in _MID_HINTS):
        return Tier.CHEAP

    params = estimate_params_b(model.model_id, model.name)
    if params is not None:
        if params <= tiering.cheap_max_params_b:
            return Tier.CHEAP
        if params <= tiering.mid_max_params_b:
            return Tier.MID
        return Tier.HARD

    if any(h in blob for h in _MID_HINTS):
        return Tier.MID
    # Unknown size: mid is the safe default (never silent cheap).
    return Tier.MID


def _is_free_openrouter(row: dict[str, Any]) -> bool:
    model_id = str(row.get("id", ""))
    if model_id.endswith(":free"):
        return True
    if model_id == "openrouter/free":
        return False  # exclude auto-router from free pool
    pricing = row.get("pricing") or {}
    prompt = _safe_float(pricing.get("prompt"), default=-1.0)
    completion = _safe_float(pricing.get("completion"), default=-1.0)
    return prompt == 0.0 and completion == 0.0


def _is_chat_model(model_id: str, row: dict[str, Any] | None = None) -> bool:
    """Keep text/chat LLMs only — drop music, image, ASR, embeddings, etc."""
    blob = model_id.lower()
    if any(bad in blob for bad in _BLOCKED_ID_SUBSTR):
        return False
    row = row or {}
    arch = row.get("architecture") or {}
    modality = str(arch.get("modality") or "").lower()
    if modality:
        # OpenRouter uses forms like "text->text" or "text+image->text"
        if "->" in modality:
            inp, out = modality.split("->", 1)
            if not any(h in out for h in _CHAT_OUTPUT_HINTS):
                return False
            if any(n in out for n in _NON_CHAT_MODALITIES) and "text" not in out:
                return False
        elif any(n == modality for n in _NON_CHAT_MODALITIES):
            return False

    outputs = arch.get("output_modalities") or row.get("output_modalities") or []
    if isinstance(outputs, list) and outputs:
        outs = {str(x).lower() for x in outputs}
        if "text" not in outs:
            return False

    inputs = arch.get("input_modalities") or row.get("input_modalities") or []
    if isinstance(inputs, list) and inputs:
        inns = {str(x).lower() for x in inputs}
        # Pure audio/image input with no text is not a chat model.
        if "text" not in inns and inns & {"audio", "image", "video"}:
            return False

    # Prefer instruct / chat-style ids when free catalog is noisy.
    name = str(row.get("name") or "").lower()
    desc = str(row.get("description") or "").lower()
    if any(w in f"{blob} {name} {desc}" for w in ("music generation", "image generation", "text-to-speech", "speech-to-text")):
        return False
    return True


def _openrouter_supports_tools(row: dict[str, Any]) -> bool:
    params = row.get("supported_parameters") or []
    if isinstance(params, list):
        lowered = {str(p).lower() for p in params}
        return "tools" in lowered or "tool_choice" in lowered
    return True


def fetch_openrouter_models(
    *,
    base_url: str,
    api_key: str,
    free_only: bool = True,
    timeout_s: float = 20.0,
) -> list[DiscoveredModel]:
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        payload = resp.json()
    rows = payload.get("data") or []
    out: list[DiscoveredModel] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id") or "").strip()
        if not model_id:
            continue
        if free_only and not _is_free_openrouter(row):
            continue
        if not _is_chat_model(model_id, row):
            continue
        pricing = row.get("pricing") or {}
        # OpenRouter prices are USD per token; convert to per-1k.
        in_cost = _safe_float(pricing.get("prompt")) * 1000.0
        out_cost = _safe_float(pricing.get("completion")) * 1000.0
        ctx = int(row.get("context_length") or (row.get("top_provider") or {}).get("context_length") or 128000)
        out.append(
            DiscoveredModel(
                model_id=model_id,
                provider=Provider.OPENROUTER,
                context_window=max(1024, ctx),
                cost_per_1k_input=in_cost,
                cost_per_1k_output=out_cost,
                supports_tool_calling=_openrouter_supports_tools(row),
                supports_structured_output=True,
                name=str(row.get("name") or model_id),
                metadata={"source": "openrouter", "description": str(row.get("description") or "")[:200]},
            )
        )
    return out


def fetch_openai_compatible_models(
    *,
    base_url: str,
    api_key: str,
    provider: Provider,
    free_only: bool = False,
    prefer_suffix: str = "",
    timeout_s: float = 20.0,
) -> list[DiscoveredModel]:
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        payload = resp.json()
    rows = payload.get("data") or []
    out: list[DiscoveredModel] = []
    for row in rows:
        if isinstance(row, str):
            model_id = row.strip()
            row_dict: dict[str, Any] = {"id": model_id}
        elif isinstance(row, dict):
            row_dict = row
            model_id = str(row.get("id") or "").strip()
        else:
            continue
        if not model_id:
            continue
        if not _is_chat_model(model_id, row_dict):
            continue
        if prefer_suffix and prefer_suffix not in model_id and free_only:
            # Soft filter — keep if no suffix matches at all later.
            pass
        # Groq / OpenAI style lists have no pricing; treat as free for demo.
        owned = str(row_dict.get("owned_by") or "")
        ctx = int(row_dict.get("context_window") or row_dict.get("context_length") or 128000)
        out.append(
            DiscoveredModel(
                model_id=model_id,
                provider=provider,
                context_window=max(1024, ctx),
                cost_per_1k_input=0.0 if free_only or provider in {Provider.GROQ, Provider.OLLAMA} else 0.001,
                cost_per_1k_output=0.0 if free_only or provider in {Provider.GROQ, Provider.OLLAMA} else 0.002,
                supports_tool_calling=True,
                supports_structured_output=True,
                name=str(row_dict.get("name") or model_id),
                metadata={"source": provider.value, "owned_by": owned},
            )
        )
    if prefer_suffix and free_only:
        preferred = [m for m in out if prefer_suffix in m.model_id]
        if preferred:
            return preferred
    return out


def fetch_from_source(source: DiscoverSource, *, api_key: str | None = None) -> list[DiscoveredModel]:
    if not source.enabled:
        return []
    key = (api_key if api_key is not None else os.environ.get(source.api_key_env, "")).strip()
    # OpenRouter models catalog is public; Groq usually needs a key.
    if source.provider == Provider.OPENROUTER:
        return fetch_openrouter_models(
            base_url=source.base_url,
            api_key=key,
            free_only=source.free_only,
        )
    if not key and source.provider not in {Provider.OLLAMA}:
        logger.warning("Skipping discover source %s — missing %s", source.name, source.api_key_env)
        return []
    return fetch_openai_compatible_models(
        base_url=source.base_url,
        api_key=key,
        provider=source.provider,
        free_only=source.free_only,
        prefer_suffix=source.prefer_suffix,
    )


def _latency_guess(tier: Tier, provider: Provider, model_id: str = "") -> float:
    # Reflect real demo timings: Groq is usually sub-second; OpenRouter free
    # endpoints are often multi-second and occasionally tens of seconds.
    base = {Tier.CHEAP: 300.0, Tier.MID: 600.0, Tier.HARD: 1200.0}[tier]
    blob = model_id.lower()
    if provider == Provider.GROQ:
        latency = base * 0.6
        if "instant" in blob:
            latency *= 0.7
        # Prefer the large gpt-oss hard seat over smaller reasoning siblings.
        if "gpt-oss" in blob and "120b" not in blob:
            latency *= 1.5
        if "qwen3" in blob:
            latency *= 1.35
        return latency
    if provider == Provider.OPENROUTER:
        return base * 4.0
    return base


def discovered_to_spec(model: DiscoveredModel, tier: Tier, *, key_prefix: str | None = None) -> ModelSpec:
    safe_id = re.sub(r"[^a-zA-Z0-9._-]+", "-", model.model_id).strip("-").lower()
    key = f"{key_prefix or model.provider.value}-{safe_id}"[:120]
    return ModelSpec(
        key=key,
        provider_model_id=model.model_id,
        provider=model.provider,
        tier=tier,
        enabled=True,
        cost_per_1k_input=model.cost_per_1k_input,
        cost_per_1k_output=model.cost_per_1k_output,
        avg_latency_ms=_latency_guess(tier, model.provider, model.model_id),
        context_window=model.context_window,
        supports_tool_calling=model.supports_tool_calling,
        supports_structured_output=model.supports_structured_output,
        capabilities=["chat", "tools"] if model.supports_tool_calling else ["chat"],
        metadata={**model.metadata, "discovered": True, "display_name": model.name},
    )


def _rank_key(model: DiscoveredModel, tier: Tier) -> tuple:
    """Prefer Groq fast chat models; demote reasoning/CoT ids outside hard."""
    free = 0 if (model.cost_per_1k_input == 0.0 and model.cost_per_1k_output == 0.0) else 1
    provider_pref = 0 if model.provider == Provider.GROQ else (1 if model.provider == Provider.OPENROUTER else 2)
    blob = model.model_id.lower()
    reasoning = any(x in blob for x in _REASONING_ID_SUBSTR)
    # Outside hard, push reasoning models to the back (or keep only as last resort).
    reasoning_pref = 0 if (tier == Tier.HARD or not reasoning) else 1
    instruct_pref = 0 if any(
        x in blob for x in ("instruct", "versatile", "instant", "chat", "llama", "gemma", "mistral")
    ) else 1
    params = estimate_params_b(model.model_id, model.name) or 50.0
    if tier == Tier.CHEAP:
        size_pref = params
    elif tier == Tier.HARD:
        size_pref = -params
    else:
        size_pref = abs(params - 40.0)
    return (reasoning_pref, provider_pref, free, instruct_pref, size_pref, -model.context_window, model.model_id)


def bucket_and_limit(
    models: list[DiscoveredModel],
    tiering: TieringConfig,
) -> list[ModelSpec]:
    buckets: dict[Tier, list[DiscoveredModel]] = {Tier.CHEAP: [], Tier.MID: [], Tier.HARD: []}
    seen_ids: set[str] = set()
    for model in models:
        if model.model_id in seen_ids:
            continue
        seen_ids.add(model.model_id)
        if tiering.prefer_free and (model.cost_per_1k_input > 0 or model.cost_per_1k_output > 0):
            # Still allow if nothing free later — filter after bucketing if needed.
            pass
        tier = assign_tier(model, tiering)
        buckets[tier].append(model)

    specs: list[ModelSpec] = []
    for tier, items in buckets.items():
        if tiering.prefer_free:
            free = [m for m in items if m.cost_per_1k_input == 0.0 and m.cost_per_1k_output == 0.0]
            if free:
                items = free
        items = sorted(items, key=lambda m: _rank_key(m, tier))
        # Cap OpenRouter seats so slow free endpoints cannot crowd out Groq.
        selected: list[DiscoveredModel] = []
        openrouter_slots = max(1, tiering.max_per_tier // 3)
        openrouter_taken = 0
        for model in items:
            if len(selected) >= max(1, tiering.max_per_tier):
                break
            if model.provider == Provider.OPENROUTER:
                if openrouter_taken >= openrouter_slots:
                    continue
                openrouter_taken += 1
            selected.append(model)
        # If caps left a tier empty, fall back to top-ranked regardless of provider.
        if not selected and items:
            selected = items[:1]
        for model in selected:
            specs.append(discovered_to_spec(model, tier))
    return specs


def pick_tier_representatives(specs: list[ModelSpec]) -> dict[str, str]:
    """One primary model id per tier for passthrough client profiles."""
    out: dict[str, str] = {}
    for tier in (Tier.CHEAP, Tier.MID, Tier.HARD):
        candidates = [s for s in specs if s.tier == tier and s.enabled]
        if not candidates:
            continue
        # Prefer groq for cheap (speed), openrouter otherwise when free.
        candidates = sorted(
            candidates,
            key=lambda s: (
                0 if s.cost_per_1k_input == 0 else 1,
                0 if (tier == Tier.CHEAP and s.provider == Provider.GROQ) else 1,
                s.avg_latency_ms,
                s.provider_model_id,
            ),
        )
        out[tier.value] = candidates[0].provider_model_id
    return out


def parse_discover_config(raw: dict[str, Any]) -> DiscoverConfig:
    discover = raw.get("discover") or {}
    tiering_raw = discover.get("tiering") or {}
    tiering = TieringConfig(
        strategy=str(tiering_raw.get("strategy", "size_then_hints")),
        cheap_max_params_b=float(tiering_raw.get("cheap_max_params_b", 20)),
        mid_max_params_b=float(tiering_raw.get("mid_max_params_b", 80)),
        max_per_tier=int(tiering_raw.get("max_per_tier", 8)),
        prefer_free=bool(tiering_raw.get("prefer_free", True)),
    )
    sources: list[DiscoverSource] = []
    for row in discover.get("providers") or []:
        provider_name = str(row.get("provider") or row.get("name") or "").strip().lower()
        if not provider_name:
            continue
        try:
            provider = Provider(provider_name)
        except ValueError:
            provider = Provider.OPENAI_COMPATIBLE
        sources.append(
            DiscoverSource(
                name=str(row.get("name") or provider_name),
                provider=provider,
                base_url=str(row.get("base_url") or "").rstrip("/"),
                api_key_env=str(row.get("api_key_env") or ""),
                free_only=bool(row.get("free_only", True)),
                prefer_suffix=str(row.get("prefer_suffix") or ""),
                enabled=bool(row.get("enabled", True)),
            )
        )
    enabled = bool(raw.get("source", "discover") != "static")
    if os.environ.get("DISPATCH_DISCOVER", "").lower() in {"0", "false", "no"}:
        enabled = False
    if os.environ.get("DISPATCH_DISCOVER", "").lower() in {"1", "true", "yes"}:
        enabled = True
    return DiscoverConfig(
        enabled=enabled and bool(sources or raw.get("discover")),
        cache_ttl_seconds=int(discover.get("cache_ttl_seconds", 3600)),
        sources=sources,
        tiering=tiering,
        fallback_models=list(raw.get("fallback_models") or raw.get("models") or []),
    )


def load_discover_config(path: str | Any) -> DiscoverConfig:
    from pathlib import Path

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"models.yaml must be a mapping: {path}")
    return parse_discover_config(data)


@dataclass
class DiscoveryCache:
    specs: list[ModelSpec]
    fetched_at: float
    version: str
    source: str  # discover | fallback


def discover_models(config: DiscoverConfig, *, api_keys: dict[str, str] | None = None) -> DiscoveryCache:
    """Fetch live catalogs and build ModelSpecs. Falls back to YAML on failure."""
    api_keys = api_keys or {}
    discovered: list[DiscoveredModel] = []
    errors: list[str] = []

    if config.enabled and config.sources:
        for source in config.sources:
            try:
                key = api_keys.get(source.api_key_env) or os.environ.get(source.api_key_env, "")
                batch = fetch_from_source(source, api_key=key)
                logger.info("Discovered %d models from %s", len(batch), source.name)
                discovered.extend(batch)
            except Exception as exc:
                msg = f"{source.name}: {exc}"
                errors.append(msg)
                logger.warning("Model discovery failed for %s: %s", source.name, exc)

    specs: list[ModelSpec] = []
    source_label = "discover"
    if discovered:
        specs = bucket_and_limit(discovered, config.tiering)
        # Ensure every tier has at least one model when possible.
        present = {s.tier for s in specs}
        if Tier.CHEAP not in present and discovered:
            specs.append(discovered_to_spec(discovered[0], Tier.CHEAP))
        if Tier.MID not in present and discovered:
            specs.append(discovered_to_spec(discovered[min(1, len(discovered) - 1)], Tier.MID))
        if Tier.HARD not in present and discovered:
            specs.append(discovered_to_spec(discovered[-1], Tier.HARD))

    if not specs:
        source_label = "fallback"
        specs = _fallback_specs(config.fallback_models)
        if errors:
            logger.warning("Using fallback models after discovery errors: %s", "; ".join(errors))

    if not specs:
        raise RuntimeError(
            "No models discovered and no fallback_models configured. "
            "Set GROQ_API_KEY / OPENROUTER_API_KEY or add fallback_models in models.yaml."
        )

    blob = yaml.safe_dump(
        [{"key": s.key, "id": s.provider_model_id, "tier": s.tier.value, "provider": s.provider.value} for s in specs],
        sort_keys=True,
    )
    return DiscoveryCache(
        specs=specs,
        fetched_at=time.time(),
        version=hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16],
        source=source_label,
    )


def _fallback_specs(rows: list[dict[str, Any]]) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("key") or row.get("id") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        try:
            provider = Provider(str(row["provider"]))
            tier = Tier(str(row["tier"]))
        except (KeyError, ValueError):
            continue
        specs.append(
            ModelSpec(
                key=key,
                provider_model_id=str(row.get("provider_model_id", key)),
                provider=provider,
                tier=tier,
                enabled=bool(row.get("enabled", True)),
                cost_per_1k_input=float(row.get("cost_per_1k_input", 0.0)),
                cost_per_1k_output=float(row.get("cost_per_1k_output", 0.0)),
                avg_latency_ms=float(row.get("avg_latency_ms", 500)),
                context_window=int(row.get("context_window", 128000)),
                supports_tool_calling=bool(row.get("supports_tool_calling", True)),
                supports_structured_output=bool(row.get("supports_structured_output", True)),
                capabilities=list(row.get("capabilities", ["chat"])),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return specs


def discover_for_upstream(
    *,
    base_url: str,
    api_key: str,
    protocol: str = "openai",
    free_only: bool = False,
    tiering: TieringConfig | None = None,
    provider: Provider = Provider.OPENAI_COMPATIBLE,
) -> dict[str, str]:
    """Fetch models from a coding tool's upstream and pick tier representatives."""
    tiering = tiering or TieringConfig(prefer_free=free_only)
    if protocol == "anthropic":
        # Anthropic has no public models list without auth nuances — use hints catalog.
        catalog = [
            DiscoveredModel("claude-3-5-haiku-latest", Provider.ANTHROPIC, name="Haiku"),
            DiscoveredModel("claude-sonnet-4-20250514", Provider.ANTHROPIC, name="Sonnet"),
            DiscoveredModel("claude-opus-4-20250514", Provider.ANTHROPIC, name="Opus"),
        ]
        specs = bucket_and_limit(catalog, tiering)
        return pick_tier_representatives(specs)

    if "openrouter.ai" in base_url:
        models = fetch_openrouter_models(base_url=base_url, api_key=api_key, free_only=free_only)
    else:
        models = fetch_openai_compatible_models(
            base_url=base_url,
            api_key=api_key,
            provider=provider,
            free_only=free_only,
        )
    specs = bucket_and_limit(models, tiering)
    picked = pick_tier_representatives(specs)
    if len(picked) < 3 and models:
        # Fill missing tiers from sorted list.
        ids = [m.model_id for m in models]
        picked.setdefault("cheap", ids[0])
        picked.setdefault("mid", ids[len(ids) // 2])
        picked.setdefault("hard", ids[-1])
    return picked
