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


# --------------------------------------------------------------------------
# Token / cost accounting
# --------------------------------------------------------------------------
# USD per 1M tokens (input, output). Approximate, from the public price list.
PRICING = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
_DEFAULT_PRICE = (3.0, 15.0)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). Used for offline/mock accounting.

    Real backends should prefer the provider's reported usage when available;
    this keeps cost dashboards populated even with the offline Mock backend.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_cost(model: Optional[str], input_tokens: int, output_tokens: int) -> float:
    in_price, out_price = PRICING.get(model or "", _DEFAULT_PRICE)
    return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price


@dataclasses.dataclass
class Usage:
    """Token/cost accounting for one completion."""

    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
