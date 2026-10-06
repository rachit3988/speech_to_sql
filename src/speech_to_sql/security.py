"""API-key authentication and per-principal rate limiting, shared by the REST API and MCP."""

from __future__ import annotations

import hashlib
import hmac
import time
from collections.abc import Mapping
from dataclasses import dataclass

from speech_to_sql.config import Settings
from speech_to_sql.errors import AuthenticationError, RateLimitedError


@dataclass(frozen=True, slots=True)
class Principal:
    id: str  # stable, non-secret identifier (key fingerprint, client IP, or "stdio")


def _fingerprint(key: bytes) -> str:
    return "key_" + hashlib.sha256(key).hexdigest()[:12]


class Authenticator:
    """Accepts ``X-API-Key: <key>`` or ``Authorization: Bearer <key>`` (the form MCP clients send)."""

    def __init__(self, settings: Settings) -> None:
        self._keys = [k.get_secret_value().encode() for k in settings.api_keys]

    @property
    def enabled(self) -> bool:
        return bool(self._keys)

    def authenticate(self, headers: Mapping[str, str], client_host: str | None) -> Principal:
        if not self._keys:
            return Principal(id=f"ip_{client_host or 'unknown'}")
        presented = headers.get("x-api-key", "")
        if not presented:
            scheme, _, token = headers.get("authorization", "").partition(" ")
            if scheme.lower() == "bearer":
                presented = token.strip()
        candidate = presented.encode()
        # Compare against every key so timing does not reveal which (if any) matched.
        matched = None
        for key in self._keys:
            if hmac.compare_digest(candidate, key):
                matched = key
        if matched is None:
            raise AuthenticationError("missing or invalid API key")
        return Principal(id=_fingerprint(matched))


class RateLimiter:
    """In-process token bucket per principal.

    Each replica enforces its own budget; for a strict global limit across replicas,
    enforce limits at the gateway/load balancer as well.
    """

    def __init__(self, per_minute: int, *, max_tracked: int = 10_000) -> None:
        self._capacity = float(max(per_minute, 1))
        self._rate = self._capacity / 60.0
        self._buckets: dict[str, tuple[float, float]] = {}
        self._max_tracked = max_tracked
        self.enabled = per_minute > 0

    def check(self, key: str) -> float | None:
        """Consume one token. Returns None if allowed, else seconds until a token frees up."""
        if not self.enabled:
            return None
        now = time.monotonic()
        tokens, updated = self._buckets.get(key, (self._capacity, now))
        tokens = min(self._capacity, tokens + (now - updated) * self._rate)
        if tokens < 1.0:
            self._buckets[key] = (tokens, now)
            return (1.0 - tokens) / self._rate
        self._buckets[key] = (tokens - 1.0, now)
        if len(self._buckets) > self._max_tracked:
            self._evict(now)
        return None

    def enforce(self, key: str) -> None:
        wait = self.check(key)
        if wait is not None:
            raise RateLimitedError(wait)

    def _evict(self, now: float) -> None:
        full_after = self._capacity / self._rate
        for k, (_, updated) in list(self._buckets.items()):
            if now - updated > full_after:
                del self._buckets[k]
