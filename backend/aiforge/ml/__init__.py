"""From-scratch code-completion ML for aiforge.

This package contains a decoder-only Transformer (RoPE + RMSNorm + own
attention), a byte-level BPE tokenizer with FIM special tokens, the PSM/SPM
fill-in-the-middle transform, a training pipeline, evaluation, inference, and
ONNX export.

**torch is import-guarded.** Importing :mod:`aiforge.ml` does NOT import torch:
only the torch-free pieces (``fim``, ``config``) are eagerly available. The
heavy modules (``model``, ``train``, ``data``, ``generate``, ``eval``,
``export_onnx``) are imported lazily via :func:`available` / direct submodule
import, so the backend and CI run without torch installed.
"""

from __future__ import annotations

import importlib
from typing import Any

from . import config, fim

__all__ = ["config", "fim", "available", "torch_available", "load_module"]


def torch_available() -> bool:
    """Return True if torch is importable in this environment."""
    try:
        import torch  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def available() -> bool:
    """Alias for :func:`torch_available` (the ML runtime needs torch)."""
    return torch_available()


def load_module(name: str) -> Any:
    """Lazily import a torch-dependent submodule (e.g. 'model', 'generate')."""
    if not torch_available():
        raise RuntimeError(
            "PyTorch is required for aiforge.ml runtime modules. "
            "Install training deps: pip install -r requirements-train.txt"
        )
    return importlib.import_module(f"{__name__}.{name}")
