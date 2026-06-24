"""Tests for the resilient LLM layer: retries, circuit breaker, failover."""

import pytest

from aiforge.llm import CompletionRequest, Message, MockLLM, ResilientBackend, collect
from aiforge.llm.resilient import CircuitBreaker


def _req():
    return CompletionRequest(messages=[Message("user", "hi")])


class Flaky:
    name = "flaky"

    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient")
        yield "ok"


class Dead:
    name = "dead"

    def complete(self, request):
        raise RuntimeError("always down")
        yield  # pragma: no cover


def test_retry_then_success():
    flaky = Flaky(fail_times=1)
    backend = ResilientBackend([flaky], max_retries=3)
    assert collect(backend.complete(_req())) == "ok"
    assert flaky.calls == 2


def test_retry_exhausted_then_failover():
    backend = ResilientBackend([Flaky(fail_times=10), MockLLM()], max_retries=1)
    out = collect(backend.complete(_req()))
    assert out
    assert backend.last_provider == "mock"


def test_all_dead_raises():
    backend = ResilientBackend([Dead()], max_retries=1)
    with pytest.raises(RuntimeError):
        collect(backend.complete(_req()))


def test_circuit_breaker_trips_and_recovers():
    cb = CircuitBreaker(threshold=2, cooldown=100.0)
    cb.record_failure()
    assert not cb.is_open
    cb.record_failure()
    assert cb.is_open and cb.state() == "open"
    cb.record_success()
    assert cb.state() == "closed" and not cb.is_open


def test_circuit_breaker_cooldown_reopens():
    cb = CircuitBreaker(threshold=1, cooldown=0.0)
    cb.record_failure()
    # cooldown 0 -> immediately half-open on next check (trial allowed).
    assert not cb.is_open


def test_breaker_skips_open_provider():
    dead = Dead()
    backend = ResilientBackend([dead, MockLLM()], max_retries=0, breaker_threshold=1)
    # First request: dead fails, mock serves.
    assert collect(backend.complete(_req()))
    # dead's breaker is now open; states reflect it.
    states = backend.breaker_states()
    assert states["dead"] in ("open", "half-open")


def test_build_backend_no_failover():
    from types import SimpleNamespace

    from aiforge.llm.resilient import build_backend

    settings = SimpleNamespace(
        enable_failover=False,
        backend="mock",
        llm_max_retries=2,
        failover_list=lambda: ["mock"],
    )
    backend = build_backend(settings, factory=lambda name, m: MockLLM())
    assert isinstance(backend, MockLLM)


def test_build_backend_failover_chain():
    from types import SimpleNamespace

    from aiforge.llm.resilient import build_backend

    settings = SimpleNamespace(
        enable_failover=True,
        backend="mock",
        llm_max_retries=1,
        failover_list=lambda: ["mock", "mock"],
    )
    backend = build_backend(settings, factory=lambda name, m: MockLLM())
    assert isinstance(backend, ResilientBackend)
    assert collect(backend.complete(_req()))
