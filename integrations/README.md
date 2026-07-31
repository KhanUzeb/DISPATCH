# Dispatch surfaces

How to use the supported surfaces: **chat demo**, **dashboard**, **MCP**, and **OpenAI/Anthropic-compatible HTTP**.

Requires `.env` with `GROQ_API_KEY` and/or `OPENROUTER_API_KEY`, and `DISPATCH_PROFILE=demo` for execute mode.

## 1. Chat demo

```powershell
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000/demo** (or `/`). Served from `src/chat.html` — do not open it as `file://`.

- Status should show `execute · demo`
- Each reply shows tier / provider / model / latency
- Leave the API key field empty unless `ROUTER_API_KEY` is set

## 1b. Dashboard

Open **http://localhost:8000/dashboard** after sending prompts from the chat demo (or `/complete` / `/v1`).

Shows recent routing & generation events: tier, route confidence, provider/model, latency breakdown, tokens, cache hits, errors. Auto-refreshes; JSON feed is `GET /telemetry/recent`.

## 2. MCP server

```powershell
uv run dispatch-mcp
```

Tools: `health`, `ready`, `route`, `complete`, `refresh_models`.

Example MCP client config (stdio) — also in `integrations/mcp.json.example`:

```json
{
  "mcpServers": {
    "dispatch": {
      "command": "uv",
      "args": ["run", "dispatch-mcp"],
      "env": {
        "DISPATCH_PROFILE": "demo"
      }
    }
  }
}
```

MCP `complete` needs execute mode (`demo`).

## 3. OpenAI / Anthropic HTTP

With the API running:

| Endpoint | Profile | Notes |
|----------|---------|--------|
| `POST /v1/chat/completions` | `demo` or `default` | OpenAI chat completions |
| `POST /v1/messages` | `anthropic` | Anthropic Messages (passthrough) |
| `GET /v1/models` | any | Lists routed model |

OpenAI-compatible client settings:

- Base URL: `http://localhost:8000/v1`
- Model: `dispatch`
- API key: optional unless `ROUTER_API_KEY` is set; for passthrough, send your upstream key (or set `DISPATCH_UPSTREAM_API_KEY`)

```powershell
curl.exe -X POST http://localhost:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"dispatch","messages":[{"role":"user","content":"summarize this in one sentence: hello"}]}'
```

## Profiles (`configs/clients.yaml`)

| Profile     | Mode        | Surface                                      |
|-------------|-------------|----------------------------------------------|
| `demo`      | execute     | Chat demo, MCP, execute-mode `/v1`           |
| `default`   | passthrough | OpenAI-compatible upstream proxy             |
| `anthropic` | passthrough | Anthropic `/v1/messages` upstream proxy      |
