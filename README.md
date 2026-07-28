# Dispatch

**Semantic LLM router** that chooses the right model for each request — cheap models for simple work, stronger models only when needed — then runs the call and tracks cost/latency.

You can use Dispatch as:

- a **universal router** for OpenCode / Codex / Claude Code / Cursor (passthrough → *their* models)
- a FastAPI service (`/route`, `/complete`)
- an **OpenAI + Anthropic compatible proxy** (`/v1/chat/completions`, `/v1/messages`)
- a local demo stack (HTML metrics + Streamlit chat)

---

## What it does

```text
Prompt
  → classify (cheap / mid / hard)
  → policy (hard constraints + lowest feasible cost)
  → execute (Groq, OpenRouter, OpenAI, Anthropic, …)
  → telemetry (dashboard / Langfuse)
```

| Without Dispatch | With Dispatch |
|---|---|
| Always call one expensive model | Easy prompts go to cheaper/faster models |
| Hand-written provider if/else rules | YAML routes + model registry |
| Hard to prove savings | Per-request tier, provider, cost, latency |

Dispatch mainly reduces **spend and latency**, not token count. The prompt still uses similar tokens; cheaper models cost less.

---

## Architecture (V2)

```text
API (/route, /complete, /v1/*)
 └── RoutingService
      ├── Classifier  (Encoder + Semantic Index + RouteStore)
      ├── Policy      (hard constraints → feasible models)
      ├── Executor    (Provider adapters)
      ├── Decision cache (in-memory, fail-open)
      └── Telemetry   (logging / Langfuse / in-memory metrics)
```

- Routes: `configs/routes.yaml`
- Models: `configs/models.yaml`
- Runtime settings: `configs/router.yaml`
- Secrets: `.env` (never commit)

Unmatched prompts fall back to **mid**, never silently to **cheap**.  
Policy never picks a model that breaks hard constraints.

---

## Run a local demo (step by step)

### Prerequisites

- Python **3.11**
- [`uv`](https://github.com/astral-sh/uv)
- Groq + OpenRouter API keys (free tier is fine)

### 1) Create the environment

```powershell
cd C:\Users\uzebk\OneDrive\Desktop\context-router
uv venv --python 3.11 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -e ".[dev,dashboard]"
Copy-Item .env.example .env
```

### 2) Add keys to `.env`

Minimum for free-tier demo:

```env
GROQ_API_KEY=your_groq_key
OPENROUTER_API_KEY=your_openrouter_key
ROUTER_USE_FAKE_ENCODER=false
```

Optional:

```env
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
ROUTER_API_KEY=          # if set, protects /route, /complete, /v1/*, metrics, dashboard
ROUTER_RATE_LIMIT_PER_MINUTE=60
```

Keep `ROUTER_USE_FAKE_ENCODER=false` and install embeddings for real semantic routing:

```powershell
uv pip install -e ".[embeddings]"
```

Fake encoder is only for offline CI — it will not reach ~80% quality.

### 3) Start the API (required)

```powershell
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Verify:

| URL | Expect |
|---|---|
| http://localhost:8000/health | `{"status":"ok"}` |
| http://localhost:8000/ready | `{"ready":true}` |
| http://localhost:8000/dashboard | Live metrics page |

### 4) Start the chat demo (optional)

Open a **second** terminal:

```powershell
cd C:\Users\uzebk\OneDrive\Desktop\context-router
.\.venv\Scripts\Activate.ps1
uv run streamlit run src/chat_demo.py
```

Chat UI usually opens at http://localhost:8501

### 5) Try a quick demo script

**Simple prompt (often cheap/mid):**

```powershell
curl.exe -X POST http://localhost:8000/complete `
  -H "Content-Type: application/json" `
  -d "{\"prompt\":\"Summarize this in one sentence: Dispatch routes LLM requests.\",\"max_output_tokens\":120}"
```

**Hard prompt (often higher tier):**

```powershell
curl.exe -X POST http://localhost:8000/complete `
  -H "Content-Type: application/json" `
  -d "{\"prompt\":\"Design a distributed system for multi-region chat with failover.\",\"max_output_tokens\":300}"
```

**Decision only (no provider call):**

```powershell
curl.exe -X POST http://localhost:8000/route `
  -H "Content-Type: application/json" `
  -d "{\"prompt\":\"Translate hello to Spanish\",\"include_diagnostics\":true}"
```

### Demo checklist

1. Ask a simple summarization question → expect cheaper tier/model  
2. Ask a hard systems/design question → expect higher tier  
3. Confirm response metadata shows `tier`, `provider`, `model`  
4. Refresh http://localhost:8000/dashboard to see events

### Tests

```powershell
uv run pytest
```

---

## Integrate with coding tools (as a **router**, not a model)

Dispatch can sit in front of **your** OpenAI / OpenRouter / Anthropic keys and only
decide *which of your models* to call:

```text
OpenCode / Codex / Claude Code / Cursor
        │  model = dispatch
        ▼
     Dispatch   ← classify → cheap | mid | hard
        │         map tier → YOUR model id
        ▼
  YOUR upstream (OpenAI, OpenRouter, Anthropic, …)
```

### 1) Start Dispatch in passthrough mode

```powershell
$env:DISPATCH_PROFILE = "opencode"   # or: codex | claude-code | default
$env:DISPATCH_UPSTREAM_BASE_URL = "https://openrouter.ai/api/v1"
# Optional overrides:
# $env:DISPATCH_MODEL_CHEAP = "openai/gpt-4o-mini"
# $env:DISPATCH_MODEL_MID   = "anthropic/claude-sonnet-4"
# $env:DISPATCH_MODEL_HARD  = "anthropic/claude-opus-4"
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Check http://localhost:8000/ready — you should see `"mode":"passthrough"` and your tier models.

Profiles live in `configs/clients.yaml`. Copy-paste tool configs are under [`integrations/`](integrations/README.md).

### 2) Point the tool at Dispatch (use YOUR provider API key)

| Tool | Setting |
|---|---|
| **OpenCode** | Merge [`integrations/opencode.json.example`](integrations/opencode.json.example) into `opencode.json`; set provider API key to your OpenRouter/OpenAI key; select model `dispatch` |
| **Codex** | `OPENAI_BASE_URL=http://localhost:8000/v1`, `OPENAI_API_KEY=<your real key>`, model `dispatch` |
| **Cursor** | Custom OpenAI provider → base `http://localhost:8000/v1`, your real key, model `dispatch` |
| **Claude Code** | `DISPATCH_PROFILE=claude-code`, then `ANTHROPIC_BASE_URL=http://localhost:8000`, `ANTHROPIC_AUTH_TOKEN=<your anthropic key>` |

Dispatch forwards the tool’s Bearer / `x-api-key` to the upstream after rewriting `model`.  
If you also set `ROUTER_API_KEY`, put that on the tool for Dispatch auth and set `DISPATCH_UPSTREAM_API_KEY` for the real provider key.

### Demo mode (Dispatch owns keys)

Leave `DISPATCH_PROFILE=demo` (default). Point tools at Dispatch with any API key and Dispatch executes via Groq/OpenRouter using keys in `.env` — useful for the portfolio demo, not for “use my models”.

### Important integration notes

- Dispatch must stay running while the tool is in use.
- Client `model` is ignored for selection — always classify + map (or policy-select in demo mode).
- Passthrough preserves tools / streaming by forwarding the raw upstream response.
- Metadata: JSON field `dispatch`, headers `X-Dispatch-Tier`, `X-Dispatch-Model`, `X-Dispatch-Profile`.

---

## Integrate (legacy one-liner)

Most coding agents speak an **OpenAI-compatible** API:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/models` | List `dispatch` + mapped/enabled models |
| `POST` | `/v1/chat/completions` | OpenAI chat proxy |
| `POST` | `/v1/messages` | Anthropic Messages (Claude Code) |

```text
OPENAI_BASE_URL = http://localhost:8000/v1
OPENAI_API_KEY  = <your upstream key, or sk-any in demo mode>
model           = dispatch
```

If `.env` has `ROUTER_API_KEY`, use that same value as `OPENAI_API_KEY` / Bearer token for `/route`, `/complete`, and `/v1/*`.

### Docker (optional)

```powershell
docker build -t dispatch .
docker run --rm -p 8000:8000 --env-file .env dispatch
```

The image defaults to `ROUTER_USE_FAKE_ENCODER=true` for a fast start.

### Verify the proxy before connecting a tool

```powershell
curl.exe http://localhost:8000/v1/chat/completions `
  -H "Authorization: Bearer sk-any-value" `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"dispatch\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one short sentence\"}],\"max_tokens\":64}"
```

If this works, your coding tool can use the same base URL.

---

## API reference

### Native endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Process alive |
| `GET` | `/ready` | Dependencies initialized |
| `GET` | `/dashboard` | HTML metrics UI |
| `GET` | `/metrics/recent` | Recent telemetry events |
| `POST` | `/route` | Classify + choose model (no execution) |
| `POST` | `/complete` | Classify + choose + execute |

---

## Configuration

| File | Role |
|---|---|
| `configs/routes.yaml` | Exemplar utterances → cheap/mid/hard |
| `configs/models.yaml` | **Discovery config** — fetches live Groq/OpenRouter (or tool) catalogs; `fallback_models` for offline |
| `configs/clients.yaml` | Per-tool profiles; `discover_models: true` pulls tier models from the tool upstream |
| `configs/router.yaml` | Thresholds, timeouts, cache TTL, telemetry |
| `.env` | API keys and runtime toggles |
| `integrations/` | Ready-to-copy OpenCode / Codex / Claude Code / Cursor configs |

Models are **not hardcoded**. Demo mode discovers free-tier models from Groq + OpenRouter at startup (falls back to a small free-tier list if offline). Passthrough profiles call the tool’s `GET /v1/models` and auto-map cheap/mid/hard. Override any tier with `DISPATCH_MODEL_CHEAP` / `_MID` / `_HARD`. Refresh with `POST /models/refresh`.


### Providers

Built-in adapters:  
`groq`, `openrouter`, `openai`, `anthropic`, `google`, `together`, `deepseek`, `fireworks`, `mistral`, `ollama`, plus generic `openai_compatible`.

Default enabled (free-tier friendly):

- Groq `llama-3.1-8b-instant` (cheap)
- Groq `llama-3.3-70b-versatile` (mid)
- OpenRouter `openrouter/free` + `openai/gpt-oss-20b:free` (hard)

> Note: OpenRouter free model IDs change often. Prefer `openrouter/free` for demos.
> Old IDs like `meta-llama/llama-3.1-8b-instruct:free` may return HTTP 404.

To enable another provider:

1. Add its key in `.env`
2. Set `enabled: true` for that model in `configs/models.yaml`
3. Restart the API

---

## Project layout

```text
configs/                 # routes, models, router settings
src/
  api.py                 # FastAPI app (Dispatch)
  openai_compat.py       # /v1 OpenAI shim
  chat_demo.py           # Streamlit chat demo
  dashboard.html         # HTML metrics page
  dashboard.py           # Optional Langfuse Streamlit dashboard
  router/                # Core routing engine
tests/
Plan.md                  # Full V2 design plan
```

---

## Design guarantees

- No silent downgrade below classified tier when constraints fail
- Cache fail-open (cache errors never break routing)
- Telemetry fail-open (observability failures never fail user requests)
- Config is the source of truth (YAML, not hard-coded registries)
- Local mode works without Qdrant; fake encoder available for quick demos

---

## Security notes

Portfolio / demo hardening (not full production security):

- Prompt length and `max_output_tokens` are enforced on `/route`, `/complete`, and `/v1/*`
- Optional `ROUTER_API_KEY` protects `/route`, `/complete`, `/v1/*`, `/metrics/recent`, and `/dashboard`
- Soft per-IP rate limit via `ROUTER_RATE_LIMIT_PER_MINUTE` (default `60`; set `0` to disable)
- Upstream provider errors are sanitized before returning to clients
- Decision cache is thread-safe with a max entry bound
- Never commit `.env`; rotate leaked keys

Still intentionally out of scope for this repo: multi-tenant auth, TLS, distributed rate limits, and hardened response caching.

---

## Status

Dispatch V2 core is implemented: typed domain objects, YAML config/registries, injectable classifier, hard-constraint-safe policy, multi-provider execution with bounded retries/fallbacks, decision cache, Langfuse-ready telemetry, OpenAI-compatible shim, HTML metrics + Streamlit chat demo, plus demo-grade auth/rate limiting.

Still planned (see `Plan.md`): labeled evaluation benchmarks, richer fallback chains, response-cache hardening, and stronger production middleware.
