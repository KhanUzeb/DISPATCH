# Using Dispatch as a universal router

Dispatch has two modes:

| Mode | What happens |
|---|---|
| `execute` (profile `demo`) | Discovers **Groq + OpenRouter free-tier** models, then executes |
| `passthrough` | Discovers models from **your tool upstream** (`GET /v1/models`), classifies, forwards |

For OpenCode / Codex / Claude Code / Cursor, use **passthrough**. Tier models are fetched from the tool — not hardcoded. Optional `DISPATCH_MODEL_*` overrides still work.

## Quick start

```powershell
# Pick a profile: opencode | codex | claude-code | default
$env:DISPATCH_PROFILE = "opencode"
$env:DISPATCH_UPSTREAM_BASE_URL = "https://openrouter.ai/api/v1"
# Models are auto-discovered from the upstream. Optional overrides:
# $env:DISPATCH_MODEL_CHEAP = "…"
# $env:DISPATCH_MODEL_MID   = "…"
# $env:DISPATCH_MODEL_HARD  = "…"
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Verify: http://localhost:8000/ready should show `"mode":"passthrough"` and your tier models.

Put your **real provider API key** on the coding tool (not in Dispatch). Dispatch forwards that Bearer / x-api-key to the upstream after rewriting `model`.

## Per-tool files

| Tool | File |
|---|---|
| OpenCode | `opencode.json.example` → copy/merge into project `opencode.json` |
| Codex | `codex.env.example` |
| Claude Code | `claude-code.env.example` (uses Anthropic `/v1/messages`) |
| Cursor | `cursor.env.example` |

## How routing works

```text
Tool  --(model=dispatch)-->  Dispatch
                               │ classify → cheap|mid|hard
                               │ map tier → your model id
                               ▼
                            Upstream (OpenAI / OpenRouter / Anthropic)
                               │
Tool  <---- response ----------┘
```

Response headers: `X-Dispatch-Tier`, `X-Dispatch-Model`, `X-Dispatch-Profile`.
