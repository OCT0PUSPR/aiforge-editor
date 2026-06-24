"""Resilience wrapper around LLM backends.

Adds, on top of a raw :class:`LLMBackend`:

- **Retries** with exponential backoff (tenacity) on transient errors, applied
  to the *first chunk* so a backend that fails fast is retried before any output
  is emitted to the client.
- A simple **circuit breaker** per provider: after N consecutive failures the
  provider is skipped for a cooldown window.
- **Provider failover**: try providers in a configured order
  (e.g. anthropic -> huggingface -> mock); the mock backend is the always-up
  floor so a request never hard-fails when offline.

The wrapper is itself an :class:`LLMBackend`, so callers are unchanged.
"""

from __future__ import annotations

import time
from typing import Callable, Iterator, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import CompletionRequest, LLMBackend


class CircuitBreaker:
    """Trips open after ``threshold`` consecutive failures; resets after cooldown."""

    def __init__(self, threshold: int = 3, cooldown: float = 30.0) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self._failures = 0
        self._opened_at: Optional[float] = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.cooldown:
            # half-open: allow a trial request
            self._opened_at = None
            self._failures = 0
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.threshold:
            self._opened_at = time.monotonic()

    def state(self) -> str:
        if self.is_open:
            return "open"
        return "half-open" if self._failures else "closed"


class ResilientBackend:
    """Wraps an ordered list of backends with retries + breaker + failover."""

    name = "resilient"

    def __init__(
        self,
        backends: List[LLMBackend],
        *,
        max_retries: int = 2,
        breaker_threshold: int = 3,
        breaker_cooldown: float = 30.0,
    ) -> None:
        if not backends:
            raise ValueError("at least one backend is required")
        self.backends = backends
        self.max_retries = max(0, max_retries)
        self._breakers = {
            b.name: CircuitBreaker(breaker_threshold, breaker_cooldown) for b in backends
        }
        self.last_provider: str = ""

    def breaker_states(self) -> dict:
        return {name: b.state() for name, b in self._breakers.items()}

    def _attempt(self, backend: LLMBackend, request: CompletionRequest) -> Iterator[str]:
        """Run one backend with retries on the *first* chunk only.

        Retrying the first chunk lets a fast-failing backend be retried before
        any partial output reaches the client; once streaming has begun we don't
        retry (that would duplicate output).
        """

        @retry(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential(multiplier=0.3, max=4.0),
            reraise=True,
        )
        def _prime() -> "tuple[Optional[str], Iterator[str]]":
            stream = backend.complete(request)
            iterator = iter(stream)
            try:
                first = next(iterator)
            except StopIteration:
                return None, iter(())
            return first, iterator

        first, iterator = _prime()
        if first is not None:
            yield first
        for chunk in iterator:
            yield chunk

    def complete(self, request: CompletionRequest) -> Iterator[str]:
        errors: List[str] = []
        for backend in self.backends:
            breaker = self._breakers[backend.name]
            if breaker.is_open:
                errors.append(f"{backend.name}: circuit open")
                continue
            try:
                produced = False
                for chunk in self._attempt(backend, request):
                    produced = True
                    yield chunk
                breaker.record_success()
                self.last_provider = backend.name
                if produced or backend is self.backends[-1]:
                    return
                # No output and not the last backend: fall through to next.
            except Exception as exc:  # provider failed; try the next one
                breaker.record_failure()
                errors.append(f"{backend.name}: {exc}")
                continue
        raise RuntimeError("all LLM backends failed: " + "; ".join(errors))


def build_backend(
    settings,
    *,
    factory: Optional[Callable[[str, Optional[str]], LLMBackend]] = None,
    model: Optional[str] = None,
) -> LLMBackend:
    """Construct the active backend (resilient chain when failover is enabled)."""
    from . import get_backend

    local_dir = getattr(settings, "local_model_dir", "") or None
    make = factory or (lambda name, m: get_backend(name, model=m, local_model_dir=local_dir))
    if not settings.enable_failover:
        return make(settings.backend, model)
    chain: List[LLMBackend] = []
    for name in settings.failover_list():
        try:
            chain.append(make(name, model))
        except Exception:
            continue
    if not chain:
        chain.append(make("mock", model))
    return ResilientBackend(
        chain,
        max_retries=settings.llm_max_retries,
    )
