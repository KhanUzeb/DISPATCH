# Architecture

Dispatch is a semantic LLM router with three surfaces: **chat demo**, **MCP**, and **OpenAI/Anthropic-compatible HTTP**. Setup and endpoints: [README.md](README.md). Surface walkthroughs: [integrations/README.md](integrations/README.md).

## Request pipeline (execute mode)

Surfaces (`/route`, `/complete`, execute-mode `/v1/chat/completions`, MCP `route`/`complete`) share `RoutingService`:

```text
prompt
  → classifier.classify()     # src/router/classifier.py
  → policy.route()            # src/router/policy.py
  → executor.execute()        # src/router/executor.py  (skipped on /route)
```

Shared infrastructure:

- `bootstrap.build_routing_service` — config, routes, model registry, clients
- `http_client` — one process-wide `httpx.AsyncClient`
- `surface` — request limits + `/ready` payload
- `logging_config` — text or JSON logs (`ROUTER_LOG_*`)

### Classify (`classifier.py`)

1. Truncate/blend the prompt so long payloads do not drown the task.
2. Embed against exemplars from `configs/routes.yaml`; score with cosine similarity, route centroids, and soft keyword overlap.
3. Accept a route if it clears its threshold, or a clear-winner margin above `accept_floor`. Near-ties prefer the cheaper tier.
4. Map the winning route to `cheap` / `mid` / `hard`. Structured-output bumps `cheap` → `mid`.
5. Unmatched prompts fall back to **`mid`** (`DEFAULT_TIER`), never silent `cheap`.

### Policy (`policy.py`)

Hard constraints filter the registry. Among feasible models:

- Prefer the **exact classified tier**.
- If empty, **upgrade only**: `cheap` → `mid` → `hard`. Never downgrade.

Then pick by objective (`lowest_cost` default, or `lowest_latency`), with sticky diversification among near-ties.

### Execute (`executor.py`)

Calls the selected provider via `ProviderRegistry`. On retryable failures, tries other feasible same-tier models (then remaining), up to `max_fallbacks`. Streaming uses `execute_stream` for OpenAI-compatible `/v1/chat/completions`.

`/route` stops after policy. `/complete` and execute-mode `/v1/chat/completions` run the executor.

## Passthrough vs execute

Configured in `configs/clients.yaml`, overridable by `DISPATCH_MODE` / `DISPATCH_PROFILE`.

| Mode | Profile examples | Behavior |
|------|------------------|----------|
| **execute** | `demo` | Classify → policy against Dispatch’s registry → Dispatch calls Groq/OpenRouter/etc. |
| **passthrough** | `default`, `anthropic` | Classify → map tier to **your** upstream model → forward the protocol payload |

Passthrough uses `RoutingService.decide_for_client()` (no `policy.route` over Dispatch providers).

### Proxies

- **`openai_compat.py`** (`POST /v1/chat/completions`) — execute or passthrough
- **`anthropic_compat.py`** (`POST /v1/messages`) — passthrough only when profile `protocol` is `anthropic`

MCP `complete` is execute-only; passthrough profiles must use the HTTP proxies.

## Live model discovery (`discover.py`)

`configs/models.yaml` holds discovery config plus offline `fallback_models`. With `source: discover`, startup/`POST /models/refresh` fetch catalogs, filter non-chat IDs, then `assign_tier` + `bucket_and_limit`:

1. Name hints (`mini`/`haiku`/`flash` → cheap; `opus`/`o1`/`reasoning` → hard; `sonnet`/`70b` → mid)
2. Else parameter-size cutoffs from YAML
3. Unknown size → **mid**
4. Cap `max_per_tier`, prefer free when configured

Passthrough profiles with empty `models` use the same heuristics against the upstream (or `DISPATCH_MODEL_*` overrides).

## Security & limits

For local demo / MCP / API use:

- `ROUTER_API_KEY` — Bearer auth on `/route`, `/complete`, `/v1/*`, `/models/refresh`
- `ROUTER_RATE_LIMIT_PER_MINUTE` — process-local sliding window per client IP
- `ROUTER_MAX_REQUEST_BYTES` — reject oversized bodies before parse
- `request_limits` in `configs/router.yaml` — max prompt chars / output tokens

## Known limitations

- Decision cache and rate limiter are process-local (no multi-worker sharing).
- Telemetry: optional Langfuse and/or logging, plus an in-memory ring — no durable built-in sink.
- One shared HTTP client per process; do not construct inline `httpx` clients on hot paths.
