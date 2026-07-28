from __future__ import annotations

import os

from ..schemas import Provider
from .anthropic import AnthropicAdapter
from .google import GoogleAdapter
from .groq import GroqAdapter
from .openai_compatible import OpenAICompatibleAdapter
from .openrouter import OpenRouterAdapter
from .registry import ProviderRegistry


DEFAULT_TIMEOUT_MS = 30000

OPENAI_COMPAT_DEFAULTS: dict[Provider, dict[str, str | bool]] = {
    Provider.OPENAI: {
        "env_key": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "default_base_url": "https://api.openai.com/v1",
        "require_api_key": True,
    },
    Provider.TOGETHER: {
        "env_key": "TOGETHER_API_KEY",
        "base_url_env": "TOGETHER_BASE_URL",
        "default_base_url": "https://api.together.xyz/v1",
        "require_api_key": True,
    },
    Provider.DEEPSEEK: {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_base_url": "https://api.deepseek.com/v1",
        "require_api_key": True,
    },
    Provider.FIREWORKS: {
        "env_key": "FIREWORKS_API_KEY",
        "base_url_env": "FIREWORKS_BASE_URL",
        "default_base_url": "https://api.fireworks.ai/inference/v1",
        "require_api_key": True,
    },
    Provider.MISTRAL: {
        "env_key": "MISTRAL_API_KEY",
        "base_url_env": "MISTRAL_BASE_URL",
        "default_base_url": "https://api.mistral.ai/v1",
        "require_api_key": True,
    },
    Provider.OLLAMA: {
        "env_key": "OLLAMA_API_KEY",
        "base_url_env": "OLLAMA_BASE_URL",
        "default_base_url": "http://localhost:11434/v1",
        "require_api_key": False,
    },
    Provider.OPENAI_COMPATIBLE: {
        "env_key": "OPENAI_COMPATIBLE_API_KEY",
        "base_url_env": "OPENAI_COMPATIBLE_BASE_URL",
        "default_base_url": "http://localhost:8001/v1",
        "require_api_key": False,
    },
}


def _timeout(timeouts_ms: dict[str, int], provider: Provider) -> int:
    return int(timeouts_ms.get(provider.value, DEFAULT_TIMEOUT_MS))


def build_provider_registry(timeouts_ms: dict[str, int] | None = None) -> ProviderRegistry:
    timeouts = timeouts_ms or {}
    providers = {
        Provider.GROQ: GroqAdapter(
            api_key=os.environ.get("GROQ_API_KEY"),
            timeout_ms=_timeout(timeouts, Provider.GROQ),
        ),
        Provider.OPENROUTER: OpenRouterAdapter(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            timeout_ms=_timeout(timeouts, Provider.OPENROUTER),
        ),
        Provider.ANTHROPIC: AnthropicAdapter(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            timeout_ms=_timeout(timeouts, Provider.ANTHROPIC),
        ),
        Provider.GOOGLE: GoogleAdapter(
            api_key=os.environ.get("GOOGLE_API_KEY"),
            timeout_ms=_timeout(timeouts, Provider.GOOGLE),
        ),
    }

    for provider, meta in OPENAI_COMPAT_DEFAULTS.items():
        providers[provider] = OpenAICompatibleAdapter(
            provider=provider,
            api_key=os.environ.get(str(meta["env_key"])) or None,
            base_url=os.environ.get(str(meta["base_url_env"]), str(meta["default_base_url"])),
            timeout_ms=_timeout(timeouts, provider),
            require_api_key=bool(meta["require_api_key"]),
        )

    return ProviderRegistry(providers=providers)
