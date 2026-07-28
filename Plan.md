# Dispatch V2: Production Semantic Routing Engine

## Summary

Transform the current scaffold into a maintainable Python routing service while preserving `/route` and `/complete`.

The implementation must:

- Keep routing decisions deterministic and testable.
- Separate semantic classification, policy selection, provider execution, caching, and telemetry.
- Support Groq and OpenRouter through one provider contract.
- Make configuration the single source of truth.
- Treat semantic-router as architectural inspiration only: use its route/exemplar model, encoder abstraction, similarity scoring, thresholds, and local indexing ideas, but do not copy its implementation.
- Preserve safe fallback behavior: never silently violate hard constraints.
- Keep Qdrant, Langfuse, and hosted providers optional for local development.

## Target Architecture

Use this dependency direction:

```text
API
 └── Application Service
      ├── Classifier
      │    ├── Encoder
      │    └── Semantic Index
      ├── Policy Engine
      │    └── Model Registry
      ├── Executor
      │    └── Provider Registry
      ├── Decision/Response Cache
      └── Telemetry
```

Lower-level modules must not import `api.py`.

### Core domain objects

Add typed objects for:

- `RouteDefinition`: route name, tier, utterances, threshold, metadata.
- `ClassificationResult`: route, tier, similarity score, threshold status, classifier version, fallback reason.
- `RoutingConstraints`: hard and soft request constraints.
- `ModelSpec`: model ID, provider, tier, pricing, latency, context window, capabilities.
- `RoutingCandidate`: model plus rejection reasons and estimated cost.
- `RoutingDecision`: selected model, classification, candidates, reason, policy version.
- `ProviderRequest`: normalized prompt/messages and generation settings.
- `ProviderResponse`: text, usage, latency, model, provider request ID.
- `RoutingError`: stable error categories for API and telemetry.

## Repository Changes

### `src/router/config.py`

Create a typed configuration loader.

Responsibilities:

- Load `.env` and YAML configuration.
- Validate routes, models, providers, thresholds, limits, and cache settings.
- Resolve paths relative to the project root.
- Fail fast for malformed required configuration.
- Provide safe defaults for local mode.
- Never expose secrets in logs.

### `src/router/schemas.py`

Create shared domain and API-independent types.

Requirements:

- Use Pydantic models or dataclasses consistently.
- Keep provider SDK response types out of these objects.
- Include configuration/version identifiers in classification and routing results.
- Make every decision serializable for API responses and telemetry.

### `src/router/routes.py`

Move route loading out of `classifier.py`.

Responsibilities:

- Parse `configs/routes.yaml`.
- Validate tier names and utterance lists.
- Support route-level thresholds.
- Assign a stable route configuration version using a hash of normalized configuration.
- Reject duplicate route names and empty utterances.

### `src/router/encoder.py`

Define the encoder interface:

- `dimension`
- `model_id`
- `encode_documents(texts)`
- `encode_queries(texts)`

Implement:

- Local Hugging Face encoder using the configured model.
- Deterministic fake encoder for tests.
- Lazy model initialization with explicit warmup support.

The encoder must validate output dimensions and expose its identity for cache/index invalidation.

### `src/router/index.py`

Define the semantic index interface:

- Build or load route embeddings.
- Query top-k utterances.
- Return route name, utterance, and similarity score.
- Support an in-memory NumPy implementation first.
- Add Qdrant as an optional implementation only after local behavior is stable.

The local index should be the default because the route corpus is small and does not require an external service.

### `src/router/classifier.py`

Redesign classification around injected `RouteStore`, `Encoder`, and `SemanticIndex`.

Behavior:

1. Validate and normalize the prompt.
2. Encode the query.
3. Query top-k route utterances.
4. Aggregate scores by route.
5. Apply route/global thresholds.
6. Return the best route with confidence and diagnostics.
7. Return a safe `mid`/abstain result when no route clears threshold.
8. Apply structured-output requirements as an explicit policy signal, not a hidden tier mutation.
9. Catch only expected encoder/index errors and convert them to stable classification failures.

Do not make the rest of the application depend directly on `semantic_router.Route` or `SemanticRouter`.

### `src/router/policy.py`

Keep the policy engine pure.

Implement this selection order:

1. Build all eligible models.
2. Reject candidates violating hard constraints:
   - context window
   - provider allow/deny list
   - required structured output
   - required tool calling
   - unavailable provider
   - explicit model restrictions
3. Estimate cost using input and expected output tokens.
4. Score remaining candidates using configurable soft objectives:
   - quality/tier requirement
   - cost
   - latency
   - provider preference
5. Select the lowest-cost valid model unless the request specifies another objective.
6. Return candidate diagnostics.
7. If no candidate is valid, return an explicit `NoFeasibleModel` result.

Never fall back to a model that violates a hard constraint.

### `src/router/models.py`

Replace the hard-coded `REGISTRY` with a typed registry loaded from `configs/models.yaml`.

Add:

- Model capability flags.
- Availability/enabled status.
- Provider model name versus internal model key.
- Pricing validity metadata.
- Quality tier.
- Context window.
- Tool-calling and structured-output support.
- Config version.

Reject duplicate IDs and invalid prices. Preserve lookup helpers through a registry object rather than module-level mutable lists.

### `src/router/providers/`

Create:

- `base.py`
- `groq.py`
- `openrouter.py`
- `registry.py`

The provider contract must normalize:

- request execution
- response text
- prompt/completion token usage
- provider request ID
- latency
- retryability
- rate-limit information
- upstream error category

Implement:

- Lazy client construction.
- Per-provider timeout.
- Bounded retries only for retryable failures.
- No retry for invalid requests or authentication failures.
- Provider fallback in the executor, not in the API route.
- OpenRouter support before declaring V2 execution complete.

### `src/router/executor.py`

Create an execution service.

Responsibilities:

- Receive a `RoutingDecision`.
- Build a normalized provider request.
- Execute the selected model.
- Retry only according to provider policy.
- Try configured fallback candidates when allowed.
- Recalculate actual cost from provider usage.
- Return a normalized `ProviderResponse`.
- Preserve the original decision and all fallback attempts for telemetry.

### `src/router/cache.py`

Redesign into explicit cache interfaces.

Implement decision caching first:

- Key by normalized prompt, request constraints, route version, encoder identity, policy version, and namespace.
- Store classification and routing decision, not only model ID.
- Enforce TTL.
- Validate payloads before use.
- Fail open on cache errors.

Implement response caching only as an opt-in feature:

- Require deterministic request settings.
- Include model, provider, prompt, generation options, policy version, and namespace in the key.
- Prevent cross-tenant reuse.
- Never cache failed or partial responses.
- Add configurable maximum response size.

Use in-memory cache for local mode and Qdrant only for semantic retrieval or distributed cache mode.

### `src/router/telemetry.py`

Replace direct tracing logic with an abstraction.

Emit:

- structured routing decision event
- classification latency
- policy latency
- provider latency
- total latency
- cache hit/miss
- selected model/provider/tier
- fallback attempts
- estimated and actual cost
- input/output token counts
- error category
- route confidence
- configuration versions

Default telemetry must be a no-op or standard logging implementation. Langfuse becomes an optional adapter.

Redact prompts, API keys, authorization headers, and sensitive metadata by default.

### `src/router/api.py`

Refactor API routes to call one application service.

Keep:

- `POST /route`
- `POST /complete`

Add:

- `GET /health`
- `GET /ready`

Requirements:

- Use application startup/lifespan for dependency initialization.
- Do not construct providers inside route handlers.
- Add request IDs.
- Return stable error schemas.
- Preserve prompt and output limits.
- Add authentication and rate-limiting extension points.
- Make `/route` optionally return classification confidence and candidate diagnostics.
- Make `/complete` return provider usage, fallback information, and normalized cost metadata.

### `src/dashboard.py`

Keep the dashboard, but update it to consume the stable telemetry schema.

Add views for:

- routing distribution
- no-match/abstention rate
- provider fallback rate
- provider error rate
- cache hit rate by cache type
- latency percentiles
- estimated versus actual cost
- route confidence distribution
- model utilization

Do not make the dashboard part of the routing runtime.

### Configuration files

Make these authoritative:

- `configs/routes.yaml`
- `configs/models.yaml`
- Add `configs/router.yaml`
- Update `.env.example`

`models.py` must not contain a second registry.

Configuration must include:

- encoder model and device
- thresholds
- route aggregation
- policy defaults
- provider timeouts/retries
- cache mode and TTL
- telemetry mode
- request limits

## Implementation Phases

### Phase 0 — Baseline and repository hygiene

Goal:

- Make the project reproducible before refactoring.

Changes:

- Fix package metadata and dependency grouping.
- Add a clean test command.
- Isolate optional dependencies.
- Verify project Git root and ignore rules.
- Add formatting, linting, and type-checking configuration.
- Document environment setup.

Definition of Done:

- Fresh environment installs successfully.
- Tests run without importing unrelated global pytest plugins.
- Local mode does not require Qdrant, Langfuse, or provider keys.

### Phase 1 — Domain contracts

Goal:

- Introduce stable typed objects without changing routing behavior.

Changes:

- Add `schemas.py`.
- Add stable error types.
- Add serialization tests.
- Preserve current public imports where practical.

Definition of Done:

- `/route` behavior remains compatible.
- Policy and classifier tests use domain objects.
- No provider SDK types leak into the domain layer.

### Phase 2 — Configuration and registries

Goal:

- Remove duplicated configuration.

Changes:

- Add `config.py`.
- Load routes and models from YAML.
- Add validation and versioning.
- Replace hard-coded model registry.

Definition of Done:

- Invalid configuration fails with actionable messages.
- Changing YAML changes runtime behavior without source edits.
- Registry tests cover duplicates, invalid tiers, invalid pricing, and missing fields.

### Phase 3 — Semantic classification

Goal:

- Own the semantic routing boundary while retaining semantic-router-inspired behavior.

Changes:

- Add encoder and index interfaces.
- Add local NumPy index.
- Move route loading to `routes.py`.
- Add route/global thresholds and diagnostics.
- Keep semantic-router as an optional implementation detail or remove it after parity tests.

Definition of Done:

- Classifier can run with a fake encoder and no model download.
- No-match behavior is tested.
- Threshold behavior is tested.
- Encoder identity and route version are available in results.

### Phase 4 — Policy engine

Goal:

- Make model selection explainable and constraint-safe.

Changes:

- Separate hard constraints from soft objectives.
- Return candidate rejection reasons.
- Add capability-aware routing.
- Replace silent cheapest-model fallback with explicit infeasible-policy behavior.

Definition of Done:

- No hard constraint is violated.
- Cost, latency, context, provider, structured-output, and tool-calling tests pass.
- Every decision has a machine-readable reason.

### Phase 5 — Provider execution

Goal:

- Complete real multi-provider execution.

Changes:

- Add provider adapters.
- Add normalized response and error contracts.
- Add retries and fallback candidates.
- Add OpenRouter execution.
- Move all provider calls out of `api.py`.

Definition of Done:

- Groq and OpenRouter pass the same provider contract tests.
- Retryable and non-retryable errors are distinguished.
- Fallback attempts are observable.
- Actual usage and cost are captured when available.

### Phase 6 — Caching

Goal:

- Add safe performance optimization.

Changes:

- Implement decision cache.
- Add versioned keys and namespaces.
- Add optional response cache.
- Add in-memory local backend and optional Qdrant backend.

Definition of Done:

- Cache failures never fail routing.
- Configuration/model/encoder changes invalidate incompatible entries.
- Response cache tests prove no cross-namespace reuse.

### Phase 7 — Observability

Goal:

- Make behavior measurable in production.

Changes:

- Add telemetry abstraction.
- Add structured logging and metrics.
- Add optional Langfuse adapter.
- Update dashboard.

Definition of Done:

- Every request has a correlation ID.
- Routing, execution, cache, and fallback events are correlated.
- Telemetry failures never fail user requests.
- Sensitive prompt data is not logged by default.

### Phase 8 — API hardening

Goal:

- Make the service safe to expose beyond localhost.

Changes:

- Add authentication hook.
- Add rate limiting.
- Add readiness checks.
- Add consistent error schema.
- Add async provider execution where appropriate.
- Add graceful startup/shutdown.

Definition of Done:

- Invalid requests receive stable 4xx responses.
- Upstream failures receive safe 5xx responses.
- Health and readiness distinguish process health from dependency readiness.
- Limits are enforced before paid provider calls.

### Phase 9 — Evaluation and performance

Goal:

- Demonstrate routing quality rather than relying on intuition.

Changes:

- Create a labeled evaluation dataset.
- Measure route accuracy, abstention rate, confusion matrix, cost, latency, and fallback rate.
- Sweep thresholds.
- Add load tests.
- Add encoder warmup and concurrency tests.

Definition of Done:

- Thresholds are justified by benchmark data.
- Performance budgets are documented.
- Regression tests prevent route-quality and cost regressions.

### Phase 10 — Documentation and portfolio quality

Goal:

- Make the project understandable and credible to engineers and hiring managers.

Changes:

- Rewrite README around architecture and reproducible results.
- Add configuration reference.
- Add operational runbook.
- Add architecture decision records.
- Add benchmark report.
- Add failure-mode documentation.
- Add example curl requests and local-only setup.

Definition of Done:

- A new engineer can run local routing without external services.
- Production dependencies and limitations are explicit.
- Benchmark results support the main claims.
- The repository clearly demonstrates systems, ML, provider, and observability engineering.

## Testing Strategy

Required test groups:

- Configuration validation tests.
- Domain serialization tests.
- Classifier tests with fake encoders and indexes.
- Similarity aggregation and threshold tests.
- Ambiguous/no-match tests.
- Structured-output and capability tests.
- Policy hard-constraint tests.
- Candidate scoring tests.
- Impossible-constraint tests.
- Provider adapter contract tests.
- Retry and timeout tests.
- Fallback-chain tests.
- Usage and cost calculation tests.
- Decision-cache and response-cache tests.
- Cache outage/fail-open tests.
- Telemetry redaction and telemetry-failure tests.
- FastAPI validation and error-schema tests.
- Health/readiness tests.
- End-to-end local-mode tests.
- Load and concurrency tests.

## Risks

- Semantic similarity does not guarantee task difficulty; labeled evaluation is mandatory.
- Placeholder pricing and latency must not be used for financial claims.
- Response caching can leak private data or replay stale results.
- Provider fallback can reduce quality while appearing reliable.
- Local embedding models increase startup time and memory usage.
- External Qdrant and Langfuse services add operational complexity without being necessary for the first production milestone.
- A single cheap/mid/hard taxonomy may eventually need capability-based routing dimensions.
- Broad exception handling currently hides root causes and must be replaced incrementally.
- The current repository appears to be nested under a broader Git root; repository boundaries must be verified before committing changes.

## Explicit Non-Goals

Do not:

- Copy semantic-router source code.
- Add dynamic LLM-generated routes in the first V2.
- Add multimodal routing in the first V2.
- Require Qdrant for local startup.
- Train a routing model before collecting labeled evaluation data.
- Add a plugin framework before provider and encoder interfaces are stable.
- Treat dashboard metrics as authoritative without validated telemetry schemas.
- Expand the API surface beyond the required operational endpoints until core behavior is stable.

## Final Acceptance Criteria

V2 is complete when:

- Local setup works without external infrastructure.
- Configuration has one source of truth.
- Semantic classification is injectable, thresholded, versioned, and testable.
- Policy selection is hard-constraint safe and explainable.
- Groq and OpenRouter share one provider contract.
- Provider retries and fallbacks are bounded and observable.
- Decision caching is version-safe and fail-open.
- Response caching is opt-in and namespace-safe.
- `/route` and `/complete` remain usable.
- Health/readiness and stable error schemas exist.
- Metrics and traces capture real outcomes.
- Tests cover normal paths and failure modes.
- Benchmark results justify routing thresholds.
- Documentation explains architecture, tradeoffs, limits, and operational behavior.
