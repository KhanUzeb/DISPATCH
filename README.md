# Dispatch

Semantic LLM router with three surfaces:

1. **Chat demo** — `GET /` and `/demo` (`src/chat.html`)
2. **Dashboard** — `GET /dashboard` (routing & generation events)
3. **OpenAI/Anthropic-compatible HTTP** — `/v1/chat/completions`, `/v1/messages`, `/v1/models`
4. **MCP server** — `dispatch-mcp` tools: `health`, `ready`, `route`, `complete`, `refresh_models`

Native `/route` and `/complete` back the demo and MCP.

```text
request → classify → policy constraints → execute
```

- Tiers: `cheap` / `mid` / `hard` from `configs/routes.yaml`
- Policy never violates hard constraints
- Unmatched prompts fall back to `mid` (never silent `cheap`)

See [ARCHITECTURE.md](ARCHITECTURE.md) for the pipeline, passthrough vs execute, discovery, and limits.  
See [integrations/README.md](integrations/README.md) for surface walkthroughs.

## Quick start

```powershell
cd C:\Users\uzebk\OneDrive\Desktop\context-router
uv venv --python 3.11 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"
Copy-Item .env.example .env
# Add GROQ_API_KEY and/or OPENROUTER_API_KEY; keep DISPATCH_PROFILE=demo
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Optional (better routing quality):

```powershell
uv pip install -e ".[embeddings]"
```

## Chat demo

With the API running, open http://localhost:8000/demo

Do not open `src/chat.html` as `file://` — it must be served by the API.

After sending prompts, open http://localhost:8000/dashboard to inspect routing and generation (tier, model, latency, errors).

## MCP

```powershell
dispatch-mcp
```

Example client config: [integrations/mcp.json.example](integrations/mcp.json.example).

## OpenAI-compatible API

```powershell
$env:DISPATCH_PROFILE = "demo"   # execute via Dispatch providers
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000
```

| Setting | Value |
|---------|--------|
| Base URL | `http://localhost:8000/v1` |
| Model | `dispatch` |
| API key | optional unless `ROUTER_API_KEY` is set |

Passthrough: `DISPATCH_PROFILE=default` (OpenAI upstream) or `anthropic` (`POST /v1/messages`).

```powershell
curl.exe -X POST http://localhost:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"dispatch","messages":[{"role":"user","content":"summarize this in one sentence: hello"}]}'
```

## Verify routing

```powershell
curl.exe -X POST http://localhost:8000/route `
  -H "Content-Type: application/json" `
  -d '{"prompt":"translate hello to spanish","include_diagnostics":true}'

curl.exe -X POST http://localhost:8000/complete `
  -H "Content-Type: application/json" `
  -d '{"prompt":"Summarize this in one sentence: Dispatch routes requests","max_output_tokens":120}'
```

## Config

| File | Purpose |
|------|---------|
| `configs/routes.yaml` | Classifier exemplars |
| `configs/models.yaml` | Discovery + offline fallbacks |
| `configs/clients.yaml` | Profiles: `demo`, `default`, `anthropic` |
| `configs/router.yaml` | Thresholds, policy, retries, limits |
| `.env` / `.env.example` | Keys, `DISPATCH_PROFILE`, rate limits |

## Endpoints

- `GET /` / `GET /demo` — chat UI
- `GET /dashboard` — routing & generation telemetry UI
- `GET /telemetry/recent` — recent events JSON for the dashboard
- `GET /health` · `GET /ready`
- `POST /models/refresh`
- `POST /route` · `POST /complete`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/messages`

## Tests

```powershell
uv run pytest
```
