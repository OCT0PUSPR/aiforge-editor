"""HuggingFace Inference API backend (secondary).

Talks to the HF Inference API over ``httpx`` (no heavy local model download).
Defaults to ``Qwen/Qwen2.5-Coder-7B-Instruct``, a capable open code model.

HF's text-generation endpoint is request/response (no token stream over the
public Inference API for all models), so we fetch the full generation and yield
it as a single chunk. The :class:`LLMBackend` contract only requires an
iterator of strings, so a one-element stream is valid.

``httpx`` is imported lazily so the app and offline tests load without it.
"""
from __future__ import annotations

import os
from typing import Iterator, Optional

from .base import CompletionRequest

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
API_URL = "https://api-inference.huggingface.co/models/{model}"


def _render_prompt(request: CompletionRequest) -> str:
    """Flatten the provider-neutral request into a single instruct prompt."""
    parts: list[str] = []
    if request.system:
        parts.append(f"<|system|>\n{request.system}")
    for msg in request.messages:
        tag = {"user": "<|user|>", "assistant": "<|assistant|>"}.get(msg.role, "<|user|>")
        parts.append(f"{tag}\n{msg.content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


class HuggingFaceBackend:
    """HuggingFace Inference API backend implementing :class:`LLMBackend`."""

    name = "huggingface"

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self._token = token or os.environ.get("HF_TOKEN")
        self._timeout = timeout

    def complete(self, request: CompletionRequest) -> Iterator[str]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - exercised only with httpx absent
            raise RuntimeError(
                "The 'httpx' package is required for the HuggingFace backend."
            ) from exc
        if not self._token:
            raise RuntimeError(
                "HF_TOKEN is not set. Export it or use AIFORGE_BACKEND=mock."
            )

        prompt = _render_prompt(request)
        headers = {"Authorization": f"Bearer {self._token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": request.max_tokens,
                "return_full_text": False,
                **({"temperature": request.temperature} if request.temperature is not None else {}),
                **({"stop": request.stop} if request.stop else {}),
            },
        }
        url = API_URL.format(model=request.model or self.model)
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        text = self._extract_text(data)
        if text:
            yield text

    @staticmethod
    def _extract_text(data) -> str:
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                return first.get("generated_text", "")
        if isinstance(data, dict):
            return data.get("generated_text", "")
        return ""
