# Dispatch

Dispatch is a semantic LLM router with four core surfaces:

- FastAPI endpoints: `/route`, `/complete`
- OpenAI-compatible proxy: `/v1/chat/completions`
- Anthropic-compatible proxy: `/v1/messages`
- MCP server tools: `health`, `ready`, `route`, `complete`, `refresh_models`

It keeps routing config-driven and safe:

```text
request -> classify -> policy constraints -> execute
```

- Classifier tiers: `cheap`, `mid`, `hard` from `configs/routes.yaml`
- Policy never violates hard constraints
- Unmatched prompts fall back to `mid` (not silently to `cheap`)

## Quick start

```powershell
cd C:\Users\uzebk\OneDrive\Desktop\context-router
uv venv --python 3.11 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"
Copy-Item .env.example .env
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Optional for better routing quality:

```powershell
uv pip install -e ".[embeddings]"
```

## Chat demo UI

With the API running, open:

http://localhost:8000/demo

It talks to `/complete` and shows the routed tier / provider / model under each reply.

## Verify core flow

```powershell
curl.exe -X POST http://localhost:8000/route `
  -H "Content-Type: application/json" `
  -d '{"prompt":"translate hello to spanish","include_diagnostics":true}'

curl.exe -X POST http://localhost:8000/complete `
  -H "Content-Type: application/json" `
  -d '{"prompt":"Summarize this in one sentence: Dispatch routes requests","max_output_tokens":120}'
```

## Run MCP server

```powershell
dispatch-mcp
```

## Pass-through mode (use your upstream models)

```powershell
$env:DISPATCH_PROFILE = "default"   # OpenAI-compatible
# or: claude-code (Anthropic protocol)
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Point tools to Dispatch:

- OpenAI-compatible base URL: `http://localhost:8000/v1`
- Model: `dispatch`
- API key: your real upstream key (forwarded), unless using explicit `DISPATCH_UPSTREAM_API_KEY`

## Config files

- `configs/routes.yaml` — routing exemplars
- `configs/models.yaml` — discovered/fallback model registry
- `configs/clients.yaml` — profiles (`demo`, `default`, `claude-code`)
- `configs/router.yaml` — thresholds, policy, retries, limits

## API surface

- `GET /` / `GET /demo` — chat UI
- `GET /health`
- `GET /ready`
- `POST /models/refresh`
- `POST /route`
- `POST /complete`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/messages`

## Tests

```powershell
uv run pytest
```
