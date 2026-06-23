"""AI features: inline completion, codebase chat, agentic edit, diff engine.

The submodules are the primary surface (``aiforge.ai.chat``,
``aiforge.ai.completion``, ``aiforge.ai.edit``, ``aiforge.ai.diff``). We
deliberately do NOT re-export the ``chat`` / ``complete`` functions at package
level, to avoid shadowing the same-named submodules.
"""
from . import chat, completion, diff, edit
from .diff import (
    DiffError,
    apply_patch,
    apply_unified_diff,
    make_unified_diff,
    parse_unified_diff,
    reverse_diff,
)
from .edit import (
    ApplyResult,
    EditProposal,
    apply_edit,
    apply_full_content,
    propose_edit,
)

__all__ = [
    "ApplyResult",
    "DiffError",
    "EditProposal",
    "apply_edit",
    "apply_full_content",
    "apply_patch",
    "apply_unified_diff",
    "chat",
    "completion",
    "diff",
    "edit",
    "make_unified_diff",
    "parse_unified_diff",
    "propose_edit",
    "reverse_diff",
]
