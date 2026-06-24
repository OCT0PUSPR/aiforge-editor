"""Pluggable LLM backends for aiforge.

Public surface:

- :class:`LLMBackend` -- the protocol every backend satisfies.
- :class:`CompletionRequest`, :class:`Message`, :class:`Usage` -- value types.
- :func:`get_backend` -- factory that resolves a backend by name, defaulting to
  the deterministic offline :class:`MockLLM` so nothing here ever requires a key.
- :class:`ResilientBackend` / :func:`build_backend` -- retries + circuit breaker
  + provider failover.
"""

from __future__ import annotations

from typing import Optional

from .base import (
    PRICING,
    CompletionRequest,
    LLMBackend,
    Message,
    Usage,
    collect,
    estimate_cost,
    estimate_tokens,
)
from .mock_backend import MockLLM
from .resilient import CircuitBreaker, ResilientBackend, build_backend

__all__ = [
    "PRICING",
    "CircuitBreaker",
    "CompletionRequest",
    "LLMBackend",
    "Message",
    "MockLLM",
    "ResilientBackend",
    "Usage",
    "build_backend",
    "collect",
    "estimate_cost",
    "estimate_tokens",
    "get_backend",
]


def get_backend(
    name: str = "mock",
    *,
    model: Optional[str] = None,
    local_model_dir: Optional[str] = None,
) -> LLMBackend:
    """Return a backend instance for ``name``.

    Recognised names: ``mock`` (default, offline), ``local`` (our from-scratch
    code model), ``anthropic``, ``huggingface`` / ``hf``. Unknown names -- and a
    ``local`` request with no usable checkpoint -- fall back to the mock backend
    so a misconfiguration degrades to "works offline" rather than crashing.
    """
    key = (name or "mock").strip().lower()
    if key == "anthropic":
        from .anthropic_backend import AnthropicBackend

        return AnthropicBackend(model=model) if model else AnthropicBackend()
    if key in ("huggingface", "hf"):
        from .hf_backend import HuggingFaceBackend

        return HuggingFaceBackend(model=model) if model else HuggingFaceBackend()
    if key == "local":
        from .local_backend import get_local_backend, local_model_exists

        if local_model_dir and local_model_exists(local_model_dir):
            return get_local_backend(local_model_dir)
        # No trained checkpoint available; degrade to mock.
        return MockLLM(model=model) if model else MockLLM()
    return MockLLM(model=model) if model else MockLLM()
