"""Per-identity rate limiting.

A token-bucket limiter keyed by a caller identity (user id or client IP). The
default is an in-process implementation; if ``AIFORGE_REDIS_URL`` is set and the
``redis`` package is installed, a Redis-backed limiter is used so limits hold
across multiple workers/replicas.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Tuple


class TokenBucketLimiter:
    """Thread-safe in-memory token bucket, refilling ``rpm`` tokens per minute."""

    def __init__(self) -> None:
        self._buckets: Dict[str, Tuple[float, float]] = {}  # key -> (tokens, ts)
        self._lock = threading.Lock()

    def allow(self, key: str, rpm: int, *, cost: float = 1.0) -> bool:
        if rpm <= 0:
            return True
        capacity = float(rpm)
        refill_per_sec = rpm / 60.0
        now = time.monotonic()
        with self._lock:
            tokens, ts = self._buckets.get(key, (capacity, now))
            tokens = min(capacity, tokens + (now - ts) * refill_per_sec)
            if tokens >= cost:
                self._buckets[key] = (tokens - cost, now)
                return True
            self._buckets[key] = (tokens, now)
            return False

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


class RedisLimiter:  # pragma: no cover - exercised only with redis available
    """Redis-backed fixed-window limiter (best-effort; falls back on error)."""

    def __init__(self, url: str) -> None:
        import redis  # type: ignore

        self._redis = redis.Redis.from_url(url)
        self._fallback = TokenBucketLimiter()

    def allow(self, key: str, rpm: int, *, cost: float = 1.0) -> bool:
        if rpm <= 0:
            return True
        try:
            window = int(time.time() // 60)
            rkey = f"rl:{key}:{window}"
            pipe = self._redis.pipeline()
            pipe.incr(rkey, int(cost))
            pipe.expire(rkey, 70)
            count, _ = pipe.execute()
            return int(count) <= rpm
        except Exception:
            return self._fallback.allow(key, rpm, cost=cost)

    def reset(self) -> None:
        self._fallback.reset()


def make_limiter(redis_url: Optional[str] = None):
    if redis_url:
        try:
            return RedisLimiter(redis_url)
        except Exception:
            pass
    return TokenBucketLimiter()
