from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from .cache import DecisionCacheKey, InMemoryDecisionCache
from .classifier import Classifier
from .clients import ClientConfig, ClientProfile
from .config import RouterRuntimeConfig
from .executor import ExecutionResult, Executor
from .logging_config import set_request_id
from .models import ModelRegistry
from .policy import route
from .schemas import ClassificationResult, ProviderRequest, RoutingConstraints, RoutingDecision, Tier


@dataclass
class RouteResponse:
    request_id: str
    decision: RoutingDecision
    cache_hit: bool
    classification_latency_ms: float
    policy_latency_ms: float


@dataclass(frozen=True)
class ClientRouteDecision:
    request_id: str
    classification: ClassificationResult
    tier: Tier
    selected_model: str
    profile: ClientProfile
    cache_hit: bool
    classification_latency_ms: float
    reason: str


class RoutingService:
    def __init__(
        self,
        classifier: Classifier,
        executor: Executor,
        model_registry: ModelRegistry,
        config: RouterRuntimeConfig,
        client_config: ClientConfig | None = None,
    ) -> None:
        self.classifier = classifier
        self.executor = executor
        self.model_registry = model_registry
        self.config = config
        self.client_config = client_config
        self.decision_cache = InMemoryDecisionCache(ttl_seconds=config.decision_cache_ttl_seconds)

    @property
    def active_profile(self) -> ClientProfile | None:
        if self.client_config is None:
            return None
        return self.client_config.active()

    def _input_tokens(self, prompt: str) -> int:
        return max(1, len(prompt) // 4)

    def decide(self, prompt: str, constraints: RoutingConstraints, expects_structured_output: bool) -> RouteResponse:
        request_id = str(uuid.uuid4())
        set_request_id(request_id)
        key = DecisionCacheKey(
            namespace="default",
            prompt=prompt,
            constraints={
                "max_cost_usd": constraints.max_cost_usd,
                "max_latency_ms": constraints.max_latency_ms,
                "min_context_window": constraints.min_context_window,
                "estimated_output_tokens": constraints.estimated_output_tokens,
                "require_structured_output": constraints.require_structured_output,
                "require_tool_calling": constraints.require_tool_calling,
            },
            route_version=self.classifier.route_store.version,
            encoder_id=self.classifier.encoder.identity,
            policy_version=self.config.policy_version,
        )

        cached = self.decision_cache.get(key)
        if isinstance(cached, RoutingDecision):
            return RouteResponse(
                request_id=request_id,
                decision=cached,
                cache_hit=True,
                classification_latency_ms=0.0,
                policy_latency_ms=0.0,
            )

        c_start = time.perf_counter()
        classification = self.classifier.classify(prompt, expects_structured_output=expects_structured_output)
        classification_latency_ms = (time.perf_counter() - c_start) * 1000.0

        p_start = time.perf_counter()
        decision = route(
            classification=classification,
            constraints=constraints,
            registry=self.model_registry,
            input_tokens=self._input_tokens(prompt),
            policy_version=self.config.policy_version,
            selection_seed=prompt,
            diversify=self.config.policy_diversify,
        )
        policy_latency_ms = (time.perf_counter() - p_start) * 1000.0
        self.decision_cache.set(key, decision)
        return RouteResponse(
            request_id=request_id,
            decision=decision,
            cache_hit=False,
            classification_latency_ms=classification_latency_ms,
            policy_latency_ms=policy_latency_ms,
        )

    async def decide_for_client(
        self,
        prompt: str,
        *,
        client: httpx.AsyncClient,
        expects_structured_output: bool = False,
        profile: ClientProfile | None = None,
        upstream_api_key: str | None = None,
    ) -> ClientRouteDecision:
        """Classify and map tier → the upstream's model id (no provider call)."""
        active = profile or self.active_profile
        if active is None:
            raise RuntimeError("No client profile configured")

        # Resolve tier models from the upstream when not hardcoded.
        await active.ensure_models(
            client=client,
            api_key=upstream_api_key or active.upstream_api_key,
        )

        request_id = str(uuid.uuid4())
        set_request_id(request_id)
        cache_key = DecisionCacheKey(
            namespace=f"client:{active.name}:{active.models.get('cheap')}:{active.models.get('mid')}:{active.models.get('hard')}",
            prompt=prompt,
            constraints={"expects_structured_output": expects_structured_output},
            route_version=self.classifier.route_store.version,
            encoder_id=self.classifier.encoder.identity,
            policy_version=f"client-{active.name}",
        )
        cached = self.decision_cache.get(cache_key)
        if isinstance(cached, ClientRouteDecision):
            return ClientRouteDecision(
                request_id=request_id,
                classification=cached.classification,
                tier=cached.tier,
                selected_model=cached.selected_model,
                profile=active,
                cache_hit=True,
                classification_latency_ms=0.0,
                reason=cached.reason,
            )

        c_start = time.perf_counter()
        classification = self.classifier.classify(prompt, expects_structured_output=expects_structured_output)
        classification_latency_ms = (time.perf_counter() - c_start) * 1000.0
        tier = classification.tier
        selected = active.model_for_tier(tier)
        if not selected:
            raise RuntimeError(f"Client profile '{active.name}' has no model for tier {tier.value}")

        reason = (
            f"passthrough:{active.name} classified={classification.route or 'fallback'} "
            f"tier={tier.value} → {selected}"
            + (" (discovered)" if active._discovered else "")
        )
        decision = ClientRouteDecision(
            request_id=request_id,
            classification=classification,
            tier=tier,
            selected_model=selected,
            profile=active,
            cache_hit=False,
            classification_latency_ms=classification_latency_ms,
            reason=reason,
        )
        self.decision_cache.set(cache_key, decision)
        return decision

    def _provider_request(
        self,
        prompt: str,
        constraints: RoutingConstraints,
        max_output_tokens: int,
        messages: list[dict[str, str]] | None,
        temperature: float,
    ) -> ProviderRequest:
        normalized_messages = messages or [{"role": "user", "content": prompt}]
        return ProviderRequest(
            prompt=prompt,
            messages=normalized_messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            structured_output=constraints.require_structured_output,
            tool_calling=constraints.require_tool_calling,
        )

    async def complete(
        self,
        prompt: str,
        constraints: RoutingConstraints,
        max_output_tokens: int,
        expects_structured_output: bool,
        messages: list[dict[str, str]] | None = None,
        temperature: float = 0.0,
    ) -> tuple[RouteResponse, ExecutionResult]:
        route_response = self.decide(prompt, constraints, expects_structured_output=expects_structured_output)
        provider_request = self._provider_request(
            prompt, constraints, max_output_tokens, messages, temperature
        )
        execution = await self.executor.execute(route_response.decision, provider_request)
        return route_response, execution

    async def complete_stream(
        self,
        prompt: str,
        constraints: RoutingConstraints,
        max_output_tokens: int,
        expects_structured_output: bool,
        messages: list[dict[str, str]] | None = None,
        temperature: float = 0.0,
    ) -> tuple[RouteResponse, ExecutionResult, AsyncIterator[str] | None]:
        route_response = self.decide(prompt, constraints, expects_structured_output=expects_structured_output)
        provider_request = self._provider_request(
            prompt, constraints, max_output_tokens, messages, temperature
        )
        execution, deltas = await self.executor.execute_stream(
            route_response.decision, provider_request
        )
        return route_response, execution, deltas
