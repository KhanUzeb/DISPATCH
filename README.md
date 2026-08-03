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

## Repository layout

```text
DISPATCH/
├── configs/
│   ├── routes.yaml      # cheap / mid / hard classifier exemplars
│   ├── models.yaml      # discovery settings + offline fallbacks
│   ├── clients.yaml     # profiles: demo, default, anthropic
│   └── router.yaml      # thresholds, policy, retries, request limits
├── integrations/
│   ├── mcp.json.example # MCP client config snippet
│   └── README.md        # surface walkthroughs
├── src/
│   ├── api.py           # FastAPI app: /route, /complete, UI, health
│   ├── mcp_server.py    # MCP stdio tools (dispatch-mcp)
│   ├── openai_compat.py # POST /v1/chat/completions
│   ├── anthropic_compat.py  # POST /v1/messages
│   ├── chat.html        # chat demo UI
│   ├── dashboard.html   # telemetry dashboard
│   └── router/
│       ├── bootstrap.py     # build RoutingService at startup
│       ├── classifier.py    # embed + score → cheap/mid/hard
│       ├── policy.py        # constraints + model pick
│       ├── executor.py      # call provider + fallbacks
│       ├── service.py       # decide / complete orchestration
│       ├── discover.py      # live Groq / OpenRouter catalogs
│       ├── clients.py       # profile + passthrough model map
│       ├── providers/       # groq, openrouter, openai, …
│       └── …                # config, schemas, cache, telemetry, …
├── ARCHITECTURE.md      # pipeline deep dive
├── .env.example         # API keys + DISPATCH_PROFILE
├── pyproject.toml       # package + dispatch-mcp entrypoint
└── LICENSE
```

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

MCP is optional. Install it only when you need the stdio server:

```bash
uv pip install -e ".[dev,mcp]"
```

---

## MCP

The HTTP API and routing library do not require the MCP package. Install the
`mcp` extra before running `dispatch-mcp`; without it, unrelated imports and
tests remain available and `dispatch-mcp` reports a clear startup error.

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

The router reuses one HTTP client per application lifecycle so provider
connections can be pooled. `max_fallbacks` means the number of additional
models tried after the initially selected model. Routing cache entries include
structured-output, optimization, and provider/model constraint inputs, so
changing those options cannot reuse a stale decision. For streamed provider
responses, usage is reported when the provider sends usage metadata; otherwise
the response marks usage as unavailable rather than treating it as zero.

---

## Roadmap: toward [semantic-router](https://github.com/aurelio-labs/semantic-router) maturity

[Aurelio Labs semantic-router](https://github.com/aurelio-labs/semantic-router) is the reference **library** for superfast semantic decisions (named routes, encoders, indexes, dynamic/tool routes). Dispatch is a **gateway**: classify → cheap/mid/hard → execute or passthrough, plus OpenAI/MCP surfaces.

### Strengths (Dispatch today)

| Strength | Why it matters |
|----------|----------------|
| **End-to-end gateway** | Not decide-only — classify, pick a model, call the provider (or forward upstream), return the answer |
| **Cost-tier routing** | Built-in `cheap` / `mid` / `hard` with upgrade-only policy and mid fallback (never silent cheap) |
| **OpenAI + Anthropic HTTP** | Drop-in ` /v1/chat/completions` and `/v1/messages` so existing SDKs and IDEs work without a custom client |
| **MCP server** | `health` / `ready` / `route` / `complete` / `refresh_models` for Cursor and other agents |
| **Execute + passthrough** | Demo mode runs Groq/OpenRouter for you; passthrough maps tiers onto *your* upstream model IDs |
| **Live model discovery** | Pulls free-tier catalogs at startup; refresh without redeploying hardcoded lists |
| **Provider fallbacks** | Retries across feasible models on timeouts/errors; streaming path for chat completions |
| **Ops basics** | Auth (`ROUTER_API_KEY`), rate limits, request size caps, structured logs, `/health` + `/ready` |
| **Visible telemetry** | Chat demo + dashboard show tier, model, latency, cache, errors — easy to debug routing live |
| **YAML-first config** | Routes, models, clients, and limits are editable without touching Python |
| **Free-tier friendly** | Defaults target Groq + OpenRouter free models for local demos |

Where semantic-router wins today: arbitrary named routes, encoder ecosystem, hybrid/index backends, threshold training, multi-modal, and library/docs polish. Use the gaps below to close that distance **without** giving up the gateway strengths above.

### Gaps to close

#### Product & API
| Gap | What to build |
|-----|----------------|
| **Library-first SDK** | `pip install dispatch` with a clean API (`Route`, `Router`, `encode`, `decide`) usable without running uvicorn |
| **Arbitrary named routes** | Beyond fixed `cheap`/`mid`/`hard` — user-defined intents (`billing`, `code`, `refuse`, …) that can map to models, tools, or handlers |
| **First-class abstain** | Optional `None` / no-match instead of always falling back to `mid` |
| **Dynamic / tool routes** | Extract slot values and emit structured tool calls from the route layer (not only tier pick) |
| **Save / load layers** | Serialize route utterances, embeddings, and thresholds to disk or object storage |

#### Routing quality
| Gap | What to build |
|-----|----------------|
| **Encoder plugins** | OpenAI, Cohere, FastEmbed, local GGUF — same interface; extras like `dispatch[openai]`, `dispatch[local]` |
| **Hybrid scoring** | Dense embeddings + sparse/BM25 (or keyword) with tunable blend |
| **Vector backends** | Local file today → optional Pinecone / Qdrant / Redis for large utterance sets and multi-process sync |
| **Threshold optimization** | Fit per-route thresholds from labeled prompts; report precision/recall |
| **Eval suite** | Public benchmark set (accuracy, latency p50/p99, cost) vs hand-labeled fixtures; CI gate |

#### Surfaces & ecosystem
| Gap | What to build |
|-----|----------------|
| **Multi-modal routes** | Image (and later audio) exemplars → route id |
| **Agent frameworks** | Drop-in LangChain / LlamaIndex / OpenAI Agents adapters |
| **Streaming-first UX** | SSE/WebSocket chat in the demo; token streaming as the default client path |
| **Docs site + notebooks** | Hosted docs, intro notebooks, “local only”, “optimize thresholds”, “custom routes” |
| **PyPI releases** | Semver tags, changelog, GitHub Actions publish; badge + install one-liner on the README |

**Practical order:** (1) library API + named routes + abstain → (2) encoder extras + eval/threshold tuning → (3) hybrid + vector backends → (4) dynamic/tool routes → (5) docs/PyPI/multi-modal.

---

## License

[MIT](LICENSE) © 2026 KhanUzeb
