"""Local LLM backend powered by our from-scratch code model.

Wraps :class:`aiforge.ml.generate.CodeCompleter` in the :class:`LLMBackend`
protocol so the trained model can power ``/api/complete`` directly -- no external
API, no key. It understands the fill-in-the-middle completion prompt produced by
:mod:`aiforge.ai.completion` (extracting the ``<prefix>``/``<suffix>`` payload)
and returns the infilled middle.

For chat/edit prompts it falls back to a short left-to-right continuation; the
local model is a *code-completion* model, so chat is better served by Anthropic
in production (the resilient chain handles that).

torch is imported lazily inside ``_ensure_completer`` so importing this module
never requires torch.
"""

from __future__ import annotations

import re
from typing import Iterator, List, Optional

from .base import CompletionRequest, Message

_PREFIX_RE = re.compile(r"<prefix>\n?(.*?)</prefix>", re.DOTALL)
_SUFFIX_RE = re.compile(r"<suffix>\n?(.*?)</suffix>", re.DOTALL)


def _last_user(messages: List[Message]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return ""


class LocalCodeBackend:
    """Serve completions from the locally-trained code model."""

    name = "local"

    def __init__(self, run_dir: str, *, device: str = "auto") -> None:
        self.run_dir = run_dir
        self.device = device
        self._completer = None  # lazily loaded

    def _ensure_completer(self):
        if self._completer is not None:
            return self._completer
        from ..ml.generate import CodeCompleter  # torch import happens here

        self._completer = CodeCompleter.load(self.run_dir, device=self.device)
        return self._completer

    def complete(self, request: CompletionRequest) -> Iterator[str]:
        completer = self._ensure_completer()
        user = _last_user(request.messages)
        prefix_match = _PREFIX_RE.search(user)
        suffix_match = _SUFFIX_RE.search(user)

        max_new = min(request.max_tokens, 96)
        if prefix_match is not None:
            prefix = prefix_match.group(1)
            suffix = suffix_match.group(1) if suffix_match else ""
            yield from completer.stream(
                prefix, suffix, max_new_tokens=max_new, temperature=0.4, top_k=40
            )
            return
        # Non-FIM prompt: continue the raw user text.
        yield from completer.stream(user, "", max_new_tokens=max_new, temperature=0.5, top_k=40)


_CACHE: dict = {}


def get_local_backend(run_dir: str, *, device: str = "auto") -> "LocalCodeBackend":
    """Return a cached local backend for ``run_dir`` (loads the model once)."""
    key = (run_dir, device)
    backend = _CACHE.get(key)
    if backend is None:
        backend = LocalCodeBackend(run_dir, device=device)
        _CACHE[key] = backend
    return backend


def local_model_exists(run_dir: Optional[str]) -> bool:
    """True if a usable local checkpoint exists at ``run_dir``."""
    if not run_dir:
        return False
    from pathlib import Path

    run = Path(run_dir)
    has_cfg = (run / "model_config.json").exists() and (run / "tokenizer.json").exists()
    has_ckpt = (run / "best.pt").exists() or (run / "ckpt.pt").exists()
    return has_cfg and has_ckpt
