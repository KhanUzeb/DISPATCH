"""Backward-compatible trace shim for legacy imports.

Prefer `router.telemetry` in new code.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

from .schemas import RoutingDecision


@dataclass
class RoutingOutcome:
    decision: RoutingDecision
    actual_latency_ms: float
    actual_cost_usd: float
    cache_hit: bool


def log_outcome(outcome: RoutingOutcome, input_tokens: int, output_tokens: int) -> None:
    return


@contextmanager
def timed():
    class _Timer:
        elapsed_ms: float = 0.0

    t = _Timer()
    start = time.perf_counter()
    yield t
    t.elapsed_ms = (time.perf_counter() - start) * 1000
