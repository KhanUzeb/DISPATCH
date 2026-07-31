"""Process-wide shared httpx.AsyncClient (connection pooling).

Created once at process startup (FastAPI lifespan or MCP bootstrap) and
closed on shutdown. Request handlers must reuse this client — never
construct httpx.AsyncClient / httpx.Client inline on hot paths.
"""

from __future__ import annotations

import httpx

_http_client: httpx.AsyncClient | None = None

# Generous default for upstream LLM calls; per-request overrides still work.
DEFAULT_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


def create_http_client(*, timeout: httpx.Timeout | float | None = None) -> httpx.AsyncClient:
    """Construct the process-wide client (call only at startup)."""
    return httpx.AsyncClient(timeout=timeout if timeout is not None else DEFAULT_TIMEOUT)


def set_http_client(client: httpx.AsyncClient) -> None:
    global _http_client
    _http_client = client


def get_http_client() -> httpx.AsyncClient:
    if _http_client is None:
        raise RuntimeError(
            "Shared httpx.AsyncClient is not initialized — "
            "create it in FastAPI lifespan or MCP startup"
        )
    return _http_client


def try_get_http_client() -> httpx.AsyncClient | None:
    return _http_client


async def aclose_http_client() -> None:
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
