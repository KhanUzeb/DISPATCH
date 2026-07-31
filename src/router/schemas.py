from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Provider(str, Enum):
    GROQ = "groq"
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    TOGETHER = "together"
    DEEPSEEK = "deepseek"
    FIREWORKS = "fireworks"
    MISTRAL = "mistral"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"


class Tier(str, Enum):
    CHEAP = "cheap"
    MID = "mid"
    HARD = "hard"


class RoutingErrorCategory(str, Enum):
    INVALID_REQUEST = "invalid_request"
    CLASSIFICATION_FAILED = "classification_failed"
    NO_FEASIBLE_MODEL = "no_feasible_model"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_AUTH = "provider_auth"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_INVALID_RESPONSE = "provider_invalid_response"
    UPSTREAM_ERROR = "upstream_error"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class RouteDefinition:
    name: str
    tier: Tier
    utterances: list[str]
    threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClassificationResult:
    route: str | None
    tier: Tier
    similarity_score: float
    passed_threshold: bool
    classifier_version: str
    route_config_version: str
    encoder_id: str
    fallback_reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingConstraints:
    max_cost_usd: float | None = None
    max_latency_ms: float | None = None
    min_context_window: int | None = None
    estimated_output_tokens: int = 500
    allow_providers: list[Provider] | None = None
    deny_providers: list[Provider] | None = None
    require_structured_output: bool = False
    require_tool_calling: bool = False
    explicit_model_allowlist: list[str] | None = None
    optimization_objective: str = "lowest_cost"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    provider_model_id: str
    provider: Provider
    tier: Tier
    enabled: bool
    cost_per_1k_input: float
    cost_per_1k_output: float
    avg_latency_ms: float
    context_window: int
    supports_tool_calling: bool
    supports_structured_output: bool
    capabilities: list[str] = field(default_factory=list)
    pricing_validated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingCandidate:
    model: ModelSpec
    rejected_reasons: list[str]
    estimated_cost_usd: float | None
    objective_score: float | None

    @property
    def is_feasible(self) -> bool:
        return not self.rejected_reasons


@dataclass(frozen=True)
class RoutingDecision:
    selected_model: ModelSpec | None
    classification: ClassificationResult
    candidates: list[RoutingCandidate]
    reason: str
    policy_version: str
    error_category: RoutingErrorCategory | None = None


@dataclass(frozen=True)
class ProviderRequest:
    prompt: str
    messages: list[dict[str, str]]
    temperature: float = 0.0
    max_output_tokens: int = 32768
    structured_output: bool = False
    tool_calling: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    usage_input_tokens: int
    usage_output_tokens: int
    latency_ms: float
    model: str
    provider: Provider
    provider_request_id: str | None = None
    retryable: bool = False
    rate_limited: bool = False
    upstream_error_category: str | None = None


@dataclass(frozen=True)
class RoutingError:
    category: RoutingErrorCategory
    message: str
    retryable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
