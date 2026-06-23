"""LLM backend protocol and shared types.

Every backend (Anthropic, HuggingFace, Mock) implements :class:`LLMBackend`.
The contract is intentionally tiny: a streaming ``complete`` that yields text
chunks, plus a ``name`` for diagnostics. Higher-level AI features
(:mod:`aiforge.ai`) build prompts and consume the stream; they never talk to a
vendor SDK directly.
"""
from __future__ import annotations

import dataclasses
from typing import Iterable, Iterator, List, Optional, Protocol, runtime_checkable


@dataclasses.dataclass
class Message:
    """A single chat message in the provider-neutral format."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclasses.dataclass
class CompletionRequest:
    """A request for a (possibly streamed) text completion.

    ``system`` is kept separate from ``messages`` because the Anthropic API
    treats the system prompt as a distinct top-level field.
    """

    messages: List[Message]
    system: Optional[str] = None
    max_tokens: int = 1024
    temperature: Optional[float] = None
    stop: Optional[List[str]] = None
    model: Optional[str] = None


@runtime_checkable
class LLMBackend(Protocol):
    """Interchangeable text-generation backend.

    Implementations MUST be safe to construct without network access. Network
    or key errors should only surface when :meth:`complete` is actually
    iterated, so the server can boot (and tests can import) regardless of
    credentials.
    """

    name: str

    def complete(self, request: CompletionRequest) -> Iterator[str]:
        """Yield text chunks for ``request``. Implementations stream."""
        ...


def collect(chunks: Iterable[str]) -> str:
    """Join a stream of chunks into a single string (test/helper convenience)."""
    return "".join(chunks)
