"""Anthropic (Claude) LLM backend.

Uses the official ``anthropic`` SDK with streaming. The default models follow
the Claude 4.x family (Opus 4.8 for the heavy agentic edit, Sonnet 4.6 for
chat, Haiku 4.5 for low-latency inline completion). Adaptive thinking is the
recommended mode for these models, but inline completion and short edits want
fast, literal output, so thinking is left off by default and can be enabled per
feature via configuration if desired.

The import of ``anthropic`` is deferred to construction time so the rest of the
app (and the offline test suite) loads even when the SDK is not installed.
"""
from __future__ import annotations

import os
from typing import Iterator, List, Optional

from .base import CompletionRequest, Message

# Model ids per feature. Sourced from the Claude model catalogue; do not append
# date suffixes to these aliases.
DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicBackend:
    """Streaming Anthropic backend implementing :class:`LLMBackend`."""

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None  # lazily constructed in complete()

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - exercised only with SDK absent
            raise RuntimeError(
                "The 'anthropic' package is required for the Anthropic backend. "
                "Install it with `pip install anthropic` or use AIFORGE_BACKEND=mock."
            ) from exc
        if not self._api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it or use AIFORGE_BACKEND=mock."
            )
        self._client = Anthropic(api_key=self._api_key)
        return self._client

    @staticmethod
    def _to_anthropic_messages(messages: List[Message]) -> List[dict]:
        # The Anthropic API only accepts user/assistant roles in `messages`;
        # the system prompt is passed separately.
        return [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]

    def complete(self, request: CompletionRequest) -> Iterator[str]:
        client = self._ensure_client()
        kwargs = {
            "model": request.model or self.model,
            "max_tokens": request.max_tokens,
            "messages": self._to_anthropic_messages(request.messages),
        }
        if request.system:
            kwargs["system"] = request.system
        if request.temperature is not None:
            # Note: sampling params are rejected on Opus 4.7+/Fable. Callers
            # that target those models should leave temperature as None.
            kwargs["temperature"] = request.temperature
        if request.stop:
            kwargs["stop_sequences"] = request.stop

        # Stream so large / slow generations don't hit HTTP timeouts.
        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text
