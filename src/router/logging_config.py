"""Process-wide logging setup for Dispatch startup and request diagnostics.

Text format (default) keeps the familiar ``[dispatch] …`` stderr lines for local
terminals. Set ``ROUTER_LOG_FORMAT=json`` for structured logs in deployed envs.
``ROUTER_LOG_LEVEL`` defaults to INFO.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any

_request_id: ContextVar[str] = ContextVar("dispatch_request_id", default="")
_configured = False


def get_request_id() -> str:
    return _request_id.get()


def set_request_id(request_id: str) -> Token[str]:
    return _request_id.set(request_id)


def reset_request_id(token: Token[str]) -> None:
    _request_id.reset(token)


def clear_request_id() -> None:
    _request_id.set("")


class RequestIdFilter(logging.Filter):
    """Attach ``request_id`` from the context var (or LogRecord.extra) to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        existing = getattr(record, "request_id", None)
        if not existing:
            record.request_id = _request_id.get() or ""
        return True


class DispatchTextFormatter(logging.Formatter):
    """Readable local-terminal lines matching the old ``[dispatch] …`` prints."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        rid = getattr(record, "request_id", "") or ""
        if rid:
            return f"[dispatch] request_id={rid} {message}"
        return f"[dispatch] {message}"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = getattr(record, "request_id", "") or ""
        if rid:
            payload["request_id"] = rid
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(*, force: bool = False) -> None:
    """Configure the ``src`` package logger once (stderr, text or JSON)."""
    global _configured
    if _configured and not force:
        return

    level_name = (os.environ.get("ROUTER_LOG_LEVEL") or "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = (os.environ.get("ROUTER_LOG_FORMAT") or "text").strip().lower()
    if fmt not in {"json", "text"}:
        fmt = "text"

    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(RequestIdFilter())
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(DispatchTextFormatter())

    package = logging.getLogger("src")
    package.handlers.clear()
    package.addHandler(handler)
    package.setLevel(level)
    # Keep uvicorn/root access logs separate; avoid duplicate dispatch lines.
    package.propagate = False

    _configured = True
