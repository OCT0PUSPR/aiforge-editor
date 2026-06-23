"""Pluggable LLM backends for aiforge.

Public surface:

- :class:`LLMBackend` -- the protocol every backend satisfies.
- :class:`CompletionRequest`, :class:`Message` -- request/value types.
- :func:`get_backend` -- factory that resolves a backend by name, defaulting to
  the deterministic offline :class:`MockLLM` so nothing here ever requires a key.
"""
from __future__ import annotations

from typing import Optional

from .base import CompletionRequest, LLMBackend, Message, collect
from .mock_backend import MockLLM

__all__ = [
    "CompletionRequest",
    "LLMBackend",
    "Message",
    "MockLLM",
    "collect",
    "get_backend",
]


def get_backend(name: str = "mock", *, model: Optional[str] = None) -> LLMBackend:
    """Return a backend instance for ``name``.

    Recognised names: ``mock`` (default, offline), ``anthropic``, ``huggingface``
    / ``hf``. Unknown names fall back to the mock backend so a misconfiguration
    degrades to "works offline" rather than crashing the server.
    """
    key = (name or "mock").strip().lower()
    if key == "anthropic":
        from .anthropic_backend import AnthropicBackend

        return AnthropicBackend(model=model) if model else AnthropicBackend()
    if key in ("huggingface", "hf"):
        from .hf_backend import HuggingFaceBackend

        return HuggingFaceBackend(model=model) if model else HuggingFaceBackend()
    return MockLLM(model=model) if model else MockLLM()
