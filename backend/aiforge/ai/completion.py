"""Inline code completion (fill-in-the-middle).

Given the text before the cursor (``prefix``) and after it (``suffix``), build a
fill-in-the-middle prompt and stream a completion from the active backend. The
prompt is shaped so :class:`MockLLM` recognises it offline, and so a real model
returns only the text to insert at the cursor (no surrounding code).
"""
from __future__ import annotations

from typing import Iterator, Optional

from ..llm.base import CompletionRequest, LLMBackend, Message

_SYSTEM = (
    "You are an expert code-completion engine performing fill-in-the-middle. "
    "Complete the code at the cursor. Return ONLY the text that should be "
    "inserted between the prefix and the suffix -- no explanations, no code "
    "fences, no repetition of the surrounding code."
)


def build_prompt(prefix: str, suffix: str, *, language: str = "", path: str = "") -> str:
    lang = f" (language: {language})" if language else ""
    file = f" in file {path}" if path else ""
    return (
        f"Complete the code at the cursor{file}{lang}.\n\n"
        f"<prefix>\n{prefix}</prefix>\n"
        f"<suffix>\n{suffix}</suffix>\n"
        "Insert the missing code at the cursor (between prefix and suffix)."
    )


def complete(
    backend: LLMBackend,
    *,
    prefix: str,
    suffix: str = "",
    language: str = "",
    path: str = "",
    max_tokens: int = 256,
    model: Optional[str] = None,
) -> Iterator[str]:
    """Stream a fill-in-the-middle completion for the cursor position."""
    request = CompletionRequest(
        system=_SYSTEM,
        messages=[Message(role="user", content=build_prompt(prefix, suffix, language=language, path=path))],
        max_tokens=max_tokens,
        stop=None,
        model=model,
    )
    yield from backend.complete(request)
