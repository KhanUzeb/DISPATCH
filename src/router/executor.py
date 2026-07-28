from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .providers.registry import ProviderRegistry
from .schemas import ModelSpec, ProviderRequest, ProviderResponse, RoutingDecision, RoutingError, RoutingErrorCategory
from .security import client_safe_upstream_message, is_retryable_provider_error


_TIMESTAMP_LYRIC_RE = re.compile(r"\[\d+(?:\.\d+)?(?::\d+(?:\.\d+)?)?\]")
_THINK_BLOCK_RE = re.compile(
    r"<think>[\s\S]*?</think>|<thinking>[\s\S]*?</thinking>|"
    r"^\s*Here's a thinking process:[\s\S]*?(?=\n(?:Here's |```|def |Below |TL;DR)|$)",
    re.IGNORECASE | re.MULTILINE,
)


def _looks_like_non_chat_output(text: str) -> bool:
    """Detect music/timestamp junk and other non-chat payloads."""
    if _TIMESTAMP_LYRIC_RE.search(text) and text.count("[") >= 3:
        return True
    return False


def _strip_reasoning_noise(text: str) -> str:
    """Remove chain-of-thought dumps that some models prepend to answers."""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    # Drop a leading bare <think> section if the model never closed the tag.
    if "<think>" in cleaned.lower():
        lower = cleaned.lower()
        start = lower.find("<think>")
        end = lower.find("</think>")
        if end == -1:
            cleaned = cleaned[:start]
        else:
            cleaned = cleaned[:start] + cleaned[end + len("</think>") :]
    return cleaned.strip()


@dataclass(frozen=True)
class ExecutionResult:
    response: ProviderResponse | None
    decision: RoutingDecision
    fallback_attempts: list[str] = field(default_factory=list)
    error: RoutingError | None = None
    actual_cost_usd: float | None = None
    used_model: ModelSpec | None = None


class Executor:
    def __init__(
        self,
        provider_registry: ProviderRegistry,
        max_fallbacks: int = 3,
        retries_by_provider: dict[str, int] | None = None,
    ) -> None:
        self.provider_registry = provider_registry
        self.max_fallbacks = max_fallbacks
        self.retries_by_provider = retries_by_provider or {}

    def _fallback_models(self, decision: RoutingDecision) -> list[ModelSpec]:
        selected = decision.selected_model
        if selected is None:
            return []
        ordered: list[ModelSpec] = [selected]
        # Prefer other same-tier feasible models, then any remaining feasible.
        # Within each group, prefer lower avg latency (Groq before OpenRouter free).
        same_tier = sorted(
            [
                c.model
                for c in decision.candidates
                if c.is_feasible and c.model.key != selected.key and c.model.tier == selected.tier
            ],
            key=lambda m: (m.avg_latency_ms, m.key),
        )
        other = sorted(
            [
                c.model
                for c in decision.candidates
                if c.is_feasible and c.model.key != selected.key and c.model.tier != selected.tier
            ],
            key=lambda m: (m.avg_latency_ms, m.key),
        )
        for model in same_tier + other:
            if model.key not in {m.key for m in ordered}:
                ordered.append(model)
            if len(ordered) > self.max_fallbacks:
                break
        return ordered

    def _execute_with_retries(self, model: ModelSpec, request: ProviderRequest):
        adapter = self.provider_registry.get(model.provider)
        if adapter is None:
            raise LookupError(f"Provider {model.provider.value} is not configured")

        retries = max(0, int(self.retries_by_provider.get(model.provider.value, 0)))
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return adapter.execute(model.provider_model_id, request)
            except Exception as exc:
                last_exc = exc
                if attempt < retries and is_retryable_provider_error(exc):
                    time.sleep(0.25 * (attempt + 1))
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def execute(self, decision: RoutingDecision, request: ProviderRequest) -> ExecutionResult:
        if decision.selected_model is None:
            return ExecutionResult(
                response=None,
                decision=decision,
                error=RoutingError(
                    category=RoutingErrorCategory.NO_FEASIBLE_MODEL,
                    message=decision.reason,
                ),
            )

        attempts: list[str] = []
        last_error: RoutingError | None = None
        for model in self._fallback_models(decision):
            try:
                provider_response = self._execute_with_retries(model, request)
            except LookupError:
                attempts.append(f"{model.key}:provider_unavailable")
                last_error = RoutingError(
                    category=RoutingErrorCategory.PROVIDER_UNAVAILABLE,
                    message=f"Provider {model.provider.value} is not configured",
                )
                continue
            except Exception as exc:
                retryable = is_retryable_provider_error(exc)
                attempts.append(f"{model.key}:upstream_error")
                last_error = RoutingError(
                    category=RoutingErrorCategory.UPSTREAM_ERROR,
                    message=client_safe_upstream_message(model.key),
                    retryable=retryable,
                )
                continue

            text = _strip_reasoning_noise((provider_response.text or "").strip())
            # Empty or clearly non-chat payloads (e.g. music timestamp lyrics) → try next model.
            if not text or _looks_like_non_chat_output(text):
                attempts.append(f"{model.key}:empty_or_invalid_response")
                last_error = RoutingError(
                    category=RoutingErrorCategory.PROVIDER_INVALID_RESPONSE,
                    message=f"Empty/invalid response from {model.key}",
                    retryable=True,
                )
                continue

            if text != (provider_response.text or "").strip():
                provider_response = ProviderResponse(
                    text=text,
                    usage_input_tokens=provider_response.usage_input_tokens,
                    usage_output_tokens=provider_response.usage_output_tokens,
                    latency_ms=provider_response.latency_ms,
                    model=provider_response.model,
                    provider=provider_response.provider,
                    provider_request_id=provider_response.provider_request_id,
                    upstream_error_category=provider_response.upstream_error_category,
                )

            actual_cost = (
                (provider_response.usage_input_tokens / 1000.0) * model.cost_per_1k_input
                + (provider_response.usage_output_tokens / 1000.0) * model.cost_per_1k_output
            )
            return ExecutionResult(
                response=provider_response,
                decision=decision,
                fallback_attempts=attempts,
                actual_cost_usd=actual_cost,
                used_model=model,
            )

        return ExecutionResult(
            response=None,
            decision=decision,
            fallback_attempts=attempts,
            error=last_error
            or RoutingError(
                category=RoutingErrorCategory.UPSTREAM_ERROR,
                message="all candidate models failed",
            ),
        )
