"""Codebase chat with RAG context.

Answers a question about the project by retrieving relevant code chunks (via the
RAG index) plus the currently open file, assembling them into a system prompt,
and streaming the model's answer. Returns the retrieved references alongside the
stream so the UI can render "jump to source" links.
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

from ..llm.base import CompletionRequest, LLMBackend, Message
from ..rag.indexer import RagIndexer, SearchResult

_SYSTEM = (
    "You are aiforge, an expert AI pair programmer embedded in a code editor. "
    "Answer the user's question about THEIR codebase using the provided code "
    "context. Be concise and precise. When you reference code, cite the file "
    "path and line numbers. Use fenced code blocks for code. If the context is "
    "insufficient, say so rather than inventing APIs."
)


def build_context(
    results: List[SearchResult],
    *,
    open_path: str = "",
    open_content: str = "",
) -> str:
    """Render retrieved chunks (and the open file) into a context block."""
    parts: List[str] = []
    if open_path and open_content:
        parts.append(f"# Currently open file: {open_path}\n```\n{open_content}\n```")
    if results:
        parts.append("# Retrieved code context")
        for r in results:
            c = r.chunk
            sym = f" ({c.symbol})" if c.symbol else ""
            parts.append(
                f"## {c.path}:{c.start_line}-{c.end_line}{sym}\n```\n{c.text}\n```"
            )
    return "\n\n".join(parts) if parts else "(no code context available)"


def chat(
    backend: LLMBackend,
    indexer: RagIndexer,
    *,
    question: str,
    open_path: str = "",
    open_content: str = "",
    history: Optional[List[Tuple[str, str]]] = None,
    top_k: int = 6,
    max_tokens: int = 1500,
    model: Optional[str] = None,
) -> Tuple[Iterator[str], List[SearchResult]]:
    """Return a streamed answer and the retrieved references.

    ``history`` is a list of ``(role, content)`` pairs for prior turns.
    """
    results = indexer.search(question, k=top_k) if question.strip() else []
    context = build_context(results, open_path=open_path, open_content=open_content)
    system = f"{_SYSTEM}\n\n{context}"

    messages: List[Message] = []
    for role, content in (history or []):
        if role in ("user", "assistant"):
            messages.append(Message(role=role, content=content))
    messages.append(Message(role="user", content=question))

    request = CompletionRequest(
        system=system,
        messages=messages,
        max_tokens=max_tokens,
        model=model,
    )
    return backend.complete(request), results
