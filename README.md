# Dispatch

**Semantic LLM router** — classify each prompt into `cheap` / `mid` / `hard`, then send it to the right model on **Groq**, **OpenRouter**, or your own upstream.

OpenAI- and Anthropic-compatible HTTP · MCP server · local chat demo & dashboard.

```text
prompt → classify → policy → execute (or passthrough)
```

| Surface | What you get |
|---------|----------------|
| **HTTP** | `POST /v1/chat/completions`, `POST /v1/messages`, `/route`, `/complete` |
| **MCP** | `dispatch-mcp` — `health`, `ready`, `route`, `complete`, `refresh_models` |
| **UI** | Chat at `/demo`, telemetry at `/dashboard` |

Unmatched prompts fall back to **mid** (never silent cheap). Policy upgrades tiers only; it never downgrades under hard constraints.

Deeper pipeline: [ARCHITECTURE.md](ARCHITECTURE.md) · client setups: [integrations/README.md](integrations/README.md)

---

## Why Dispatch

- **Cost-aware routing** — short/simple work stays on cheap models; design & reasoning go to hard
- **Drop-in API** — point any OpenAI SDK at `http://localhost:8000/v1` with model `dispatch`
- **Live discovery** — pulls free-tier catalogs from Groq + OpenRouter at startup
- **Two modes** — **execute** (Dispatch calls providers) or **passthrough** (map tier → your upstream IDs)
- **Agent-ready** — MCP stdio tools for Cursor and other MCP clients

---

## Quick start

```bash
git clone https://github.com/KhanUzeb/DISPATCH.git
cd DISPATCH
uv venv --python 3.11 .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env   # Windows: Copy-Item .env.example .env
```

Add `GROQ_API_KEY` and/or `OPENROUTER_API_KEY` to `.env`. Keep `DISPATCH_PROFILE=demo` for execute mode.

```bash
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000/demo** (do not open `src/chat.html` as a file).  
Dashboard: **http://localhost:8000/dashboard**

Better routing quality (real embeddings):

```bash
uv pip install -e ".[embeddings]"
```

---

## MCP

```bash
uv run dispatch-mcp
```

Client example: [integrations/mcp.json.example](integrations/mcp.json.example)

```json
{
  "mcpServers": {
    "dispatch": {
      "command": "uv",
      "args": ["run", "dispatch-mcp"],
      "env": { "DISPATCH_PROFILE": "demo" }
    }
  }
}
```

---

## OpenAI-compatible API

```bash
export DISPATCH_PROFILE=demo   # Windows PowerShell: $env:DISPATCH_PROFILE = "demo"
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000
```

| Setting | Value |
|---------|--------|
| Base URL | `http://localhost:8000/v1` |
| Model | `dispatch` |
| API key | optional unless `ROUTER_API_KEY` is set |

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"dispatch","messages":[{"role":"user","content":"summarize this in one sentence: hello"}]}'
```

Passthrough: `DISPATCH_PROFILE=default` (OpenAI-compatible upstream) or `anthropic` (`POST /v1/messages`).

---

## Verify routing

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"prompt":"translate hello to spanish","include_diagnostics":true}'

curl -X POST http://localhost:8000/complete \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Summarize this in one sentence: Dispatch routes requests","max_output_tokens":120}'
```

| Prompt shape | Typical tier |
|--------------|--------------|
| Translate / summarize / extract | `cheap` |
| Explain code / write a short function | `mid` |
| Distributed design / deep tradeoffs | `hard` |

---

## Config

| File | Purpose |
|------|---------|
| `configs/routes.yaml` | Classifier exemplars for cheap / mid / hard |
| `configs/models.yaml` | Discovery + offline fallbacks |
| `configs/clients.yaml` | Profiles: `demo`, `default`, `anthropic` |
| `configs/router.yaml` | Thresholds, policy, retries, request limits |
| `.env` | API keys, `DISPATCH_PROFILE`, rate limits |

Defaults allow large completions (up to Groq-style caps) and long prompts — see `request_limits` in `configs/router.yaml`.

---

## Endpoints

| Method | Path | Role |
|--------|------|------|
| GET | `/` `/demo` | Chat UI |
| GET | `/dashboard` | Routing & generation telemetry |
| GET | `/telemetry/recent` | Recent events JSON |
| GET | `/health` `/ready` | Liveness / readiness |
| POST | `/models/refresh` | Refresh discovered models |
| POST | `/route` `/complete` | Classify only / classify + generate |
| GET | `/v1/models` | OpenAI-style model list |
| POST | `/v1/chat/completions` | OpenAI chat completions |
| POST | `/v1/messages` | Anthropic Messages (passthrough) |

---

## Tests

```bash
uv run pytest
```

---

## License

See repository license / terms as published on GitHub.
