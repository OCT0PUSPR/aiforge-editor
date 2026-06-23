"""Agentic edit: instruction -> proposed diff -> apply.

The flow has two phases so the user can review before anything is written:

1. :func:`propose_edit` -- given a natural-language instruction and a target
   file, ask the model for the full new content, then compute a unified diff
   from the current content. (Computing the diff ourselves is more robust than
   trusting the model to emit a perfectly formed diff, while still giving the
   UI a clean diff to preview.)
2. :func:`apply_edit` -- apply a previously proposed diff to the workspace,
   writing the new content. Returns the new content and the reverse diff (for
   undo).

Both phases are exercised offline by :class:`MockLLM`, which deterministically
prepends a header comment so the diff is small and reviewable.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

from ..llm.base import CompletionRequest, LLMBackend, Message, collect
from ..workspace.files import Workspace
from .diff import (
    DiffError,
    apply_unified_diff,
    make_unified_diff,
    parse_unified_diff,
    reverse_diff,
)

_SYSTEM = (
    "You are aiforge, an expert software engineer performing a precise code "
    "edit. You are given a file's current content and an instruction. Respond "
    "with the COMPLETE new content of the file after applying the instruction. "
    "Do not include explanations, markdown fences, or commentary -- output only "
    "the raw file content."
)


@dataclasses.dataclass
class EditProposal:
    path: str
    original: str
    new_content: str
    diff: str

    @property
    def changed(self) -> bool:
        return self.original != self.new_content

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "diff": self.diff,
            "new_content": self.new_content,
            "changed": self.changed,
        }


def build_prompt(path: str, content: str, instruction: str) -> str:
    return (
        f"Apply this instruction to the file.\n\n"
        f"<path>{path}</path>\n"
        f"<instruction>\n{instruction}\n</instruction>\n"
        f"<current content>\n{content}</current content>\n"
        "Return the complete new file content."
    )


def _strip_fences(text: str) -> str:
    """Remove a leading/trailing markdown code fence if the model added one."""
    stripped = text.strip("\n")
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Drop the opening fence (possibly with a language tag).
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return text


def propose_edit(
    backend: LLMBackend,
    workspace: Workspace,
    *,
    path: str,
    instruction: str,
    max_tokens: int = 4096,
    model: Optional[str] = None,
) -> EditProposal:
    """Ask the model for new content and return a reviewable proposal."""
    original = workspace.read(path) if workspace.exists(path) else ""
    request = CompletionRequest(
        system=_SYSTEM,
        messages=[Message(role="user", content=build_prompt(path, original, instruction))],
        max_tokens=max_tokens,
        model=model,
    )
    new_content = _strip_fences(collect(backend.complete(request)))
    diff = make_unified_diff(path, original, new_content)
    return EditProposal(path=path, original=original, new_content=new_content, diff=diff)


@dataclasses.dataclass
class ApplyResult:
    path: str
    new_content: str
    reverse_diff: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "new_content": self.new_content,
            "reverse_diff": self.reverse_diff,
        }


def apply_edit(
    workspace: Workspace,
    *,
    path: str,
    diff: str,
    expected_original: Optional[str] = None,
) -> ApplyResult:
    """Apply ``diff`` to ``path`` in the workspace and persist the result.

    If ``expected_original`` is provided it is used as the base text (a staleness
    guard: the diff was computed against this snapshot). Otherwise the current
    on-disk content is used.
    """
    patches = parse_unified_diff(diff)
    patch = patches[0]
    if patch.is_creation:
        original = ""
    else:
        original = (
            expected_original
            if expected_original is not None
            else (workspace.read(path) if workspace.exists(path) else "")
        )
    try:
        new_content = apply_unified_diff(original, diff)
    except DiffError as exc:
        raise DiffError(f"failed to apply edit to {path}: {exc}") from exc
    workspace.write(path, new_content)
    return ApplyResult(path=path, new_content=new_content, reverse_diff=reverse_diff(diff))


def apply_full_content(
    workspace: Workspace,
    *,
    path: str,
    new_content: str,
) -> ApplyResult:
    """Persist ``new_content`` directly and return the reverse diff for undo."""
    original = workspace.read(path) if workspace.exists(path) else ""
    workspace.write(path, new_content)
    fwd = make_unified_diff(path, original, new_content)
    return ApplyResult(path=path, new_content=new_content, reverse_diff=reverse_diff(fwd))
