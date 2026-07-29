# Using Dispatch in passthrough mode

Use the `default` profile for OpenAI-compatible tools and `claude-code` for Anthropic `/v1/messages`.

```powershell
$env:DISPATCH_PROFILE = "default"   # or claude-code
$env:DISPATCH_UPSTREAM_BASE_URL = "https://openrouter.ai/api/v1"
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Then point your tool to `http://localhost:8000/v1` and use model `dispatch`.
Dispatch classifies each request into `cheap|mid|hard`, maps tier to discovered upstream models, and forwards using your tool-provided API key.
