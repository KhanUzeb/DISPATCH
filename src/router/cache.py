from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from .schemas import RoutingDecision, to_dict


@dataclass(frozen=True)
class DecisionCacheKey:
    namespace: str
    prompt: str
    constraints: dict
    route_version: str
    encoder_id: str
    policy_version: str

    def to_hash(self) -> str:
        payload = {
            "namespace": self.namespace,
            "prompt": self.prompt.strip(),
            "constraints": self.constraints,
            "route_version": self.route_version,
            "encoder_id": self.encoder_id,
            "policy_version": self.policy_version,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class DecisionCacheEntry:
    decision: RoutingDecision
    expires_at: float


class DecisionCache(Protocol):
    def get(self, key: DecisionCacheKey) -> RoutingDecision | None:
        ...

    def set(self, key: DecisionCacheKey, decision: RoutingDecision) -> None:
        ...


class InMemoryDecisionCache:
    def __init__(self, ttl_seconds: int = 300, max_entries: int = 2048) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max(1, max_entries)
        self._store: dict[str, DecisionCacheEntry] = {}
        self._lock = threading.Lock()

    def get(self, key: DecisionCacheKey) -> RoutingDecision | None:
        try:
            hashed = key.to_hash()
            with self._lock:
                found = self._store.get(hashed)
                if not found:
                    return None
                if found.expires_at <= time.time():
                    self._store.pop(hashed, None)
                    return None
                decision = found.decision
            # Validate serializability outside the lock.
            json.dumps(to_dict(decision), sort_keys=True)
            return decision
        except Exception:
            return None

    def set(self, key: DecisionCacheKey, decision: RoutingDecision) -> None:
        try:
            json.dumps(to_dict(decision), sort_keys=True)
            hashed = key.to_hash()
            entry = DecisionCacheEntry(
                decision=decision,
                expires_at=time.time() + self.ttl_seconds,
            )
            with self._lock:
                self._evict_locked()
                self._store[hashed] = entry
        except Exception:
            return

    def _evict_locked(self) -> None:
        now = time.time()
        expired = [k for k, v in self._store.items() if v.expires_at <= now]
        for k in expired:
            self._store.pop(k, None)
        while len(self._store) >= self.max_entries:
            # Drop soonest-expiring entry (approximate LRU for TTL caches).
            oldest_key = min(self._store.items(), key=lambda item: item[1].expires_at)[0]
            self._store.pop(oldest_key, None)
