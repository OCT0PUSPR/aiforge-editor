"""AI features: inline completion, codebase chat, agentic edit, diff engine.

The submodules are the primary surface (``aiforge.ai.chat``,
``aiforge.ai.completion``, ``aiforge.ai.edit``, ``aiforge.ai.diff``). We
deliberately do NOT re-export the ``chat`` / ``complete`` functions at package
level, to avoid shadowing the same-named submodules.
"""

from . import chat, completion, diff, edit
from .diff import (
    DiffConflict,
    DiffError,
    apply_multifile,
    apply_patch,
    apply_unified_diff,
    make_multifile_diff,
    make_unified_diff,
    parse_unified_diff,
    reverse_diff,
)
from .edit import (
    ApplyResult,
    EditProposal,
    FileChange,
    MultiFileApplyResult,
    MultiFileProposal,
    apply_edit,
    apply_full_content,
    apply_multifile_edit,
    propose_edit,
    propose_multifile_edit,
)

__all__ = [
    "ApplyResult",
    "DiffConflict",
    "DiffError",
    "EditProposal",
    "FileChange",
    "MultiFileApplyResult",
    "MultiFileProposal",
    "apply_edit",
    "apply_full_content",
    "apply_multifile",
    "apply_multifile_edit",
    "apply_patch",
    "apply_unified_diff",
    "chat",
    "completion",
    "diff",
    "edit",
    "make_multifile_diff",
    "make_unified_diff",
    "parse_unified_diff",
    "propose_edit",
    "propose_multifile_edit",
    "reverse_diff",
]
