"""Lightweight auth + rate limiting for local demo / MCP / API deployments.

Not a full production security stack — enough to avoid an open proxy when
`ROUTER_API_KEY` is set, and to bound abuse on a publicly reachable port.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from collections import OrderedDict, deque

from fastapi import Header, HTTPException


def router_api_key() -> str | None:
    value = os.environ.get("ROUTER_API_KEY", "").strip()
    return value or None


def router_api_key_configured() -> bool:
    return router_api_key() is not None


def verify_router_api_key(
    authorization: str | None,
    x_api_key: str | None = None,
) -> None:
    """Require Bearer or x-api-key when ROUTER_API_KEY is configured; otherwise no-op."""
    expected = router_api_key()
    if expected is None:
        return
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
    elif x_api_key:
        token = x_api_key.strip()
    if not token or not secrets.compare_digest(token, expected):
        raise HTTPException(
            status_code=401,
            detail={
                "error_category": "auth_error",
                "message": "Missing or invalid API key",
            },
        )


def require_router_auth(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> None:
    verify_router_api_key(authorization, x_api_key=x_api_key)


def client_safe_upstream_message(model_key: str) -> str:
    """Stable client message — do not leak raw provider exception bodies."""
    return f"Upstream call failed for {model_key}"


def is_retryable_provider_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    needles = ("rate limited", "timeout", "timed out", "429", "503", "502", "connection reset")
    return any(n in msg for n in needles)


class SlidingWindowRateLimiter:
    """Simple per-key sliding window. Fail-open if misconfigured (limit <= 0).

    Tracked keys are bounded by an LRU (`max_tracked_clients`) so one-shot /
    rotating client IPs cannot grow `_hits` without bound.
    """

    def __init__(
        self,
        limit_per_minute: int = 60,
        max_tracked_clients: int = 10_000,
    ) -> None:
        self.limit_per_minute = limit_per_minute
        self.max_tracked_clients = max(1, max_tracked_clients)
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "SlidingWindowRateLimiter":
        raw = os.environ.get("ROUTER_RATE_LIMIT_PER_MINUTE", "60").strip()
        try:
            limit = int(raw)
        except ValueError:
            limit = 60
        raw_max = os.environ.get("ROUTER_RATE_LIMIT_MAX_TRACKED_CLIENTS", "10000").strip()
        try:
            max_tracked = int(raw_max)
        except ValueError:
            max_tracked = 10_000
        return cls(limit_per_minute=limit, max_tracked_clients=max_tracked)

    def allow(self, key: str) -> bool:
        if self.limit_per_minute <= 0:
            return True
        now = time.monotonic()
        window_start = now - 60.0
        with self._lock:
            bucket = self._hits.get(key)
            if bucket is None:
                bucket = deque()
                self._hits[key] = bucket
            else:
                self._hits.move_to_end(key)
            while bucket and bucket[0] < window_start:
                bucket.popleft()
            if len(bucket) >= self.limit_per_minute:
                return False
            bucket.append(now)
            while len(self._hits) > self.max_tracked_clients:
                self._hits.popitem(last=False)
            return True
