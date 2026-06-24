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
import json
import re
from typing import Dict, List, Optional

from ..llm.base import (
    CompletionRequest,
    LLMBackend,
    Message,
    Usage,
    collect,
    estimate_cost,
    estimate_tokens,
)
from ..workspace.files import Workspace
from .diff import (
    DiffConflict,
    DiffError,
    apply_multifile,
    apply_unified_diff,
    make_multifile_diff,
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
    usage: Optional[Usage] = None

    @property
    def changed(self) -> bool:
        return self.original != self.new_content

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "diff": self.diff,
            "new_content": self.new_content,
            "changed": self.changed,
            "usage": self.usage.to_dict() if self.usage else None,
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
    prompt = build_prompt(path, original, instruction)
    request = CompletionRequest(
        system=_SYSTEM,
        messages=[Message(role="user", content=prompt)],
        max_tokens=max_tokens,
        model=model,
    )
    new_content = _strip_fences(collect(backend.complete(request)))
    diff = make_unified_diff(path, original, new_content)
    usage = _account(backend, model, _SYSTEM + prompt, new_content)
    return EditProposal(
        path=path, original=original, new_content=new_content, diff=diff, usage=usage
    )


def _account(backend: LLMBackend, model: Optional[str], prompt: str, output: str) -> Usage:
    """Build a token/cost :class:`Usage` record from prompt + output text."""
    in_tok = estimate_tokens(prompt)
    out_tok = estimate_tokens(output)
    return Usage(
        provider=getattr(backend, "name", ""),
        model=model or "",
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=estimate_cost(model, in_tok, out_tok),
    )


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


# --------------------------------------------------------------------------
# Multi-file agentic edit
# --------------------------------------------------------------------------
_MULTIFILE_SYSTEM = (
    "You are aiforge, an expert software engineer performing a multi-file edit. "
    "You are given several files and an instruction. Respond with a JSON object "
    "mapping each changed file path to its COMPLETE new content, e.g. "
    '{"files": {"src/a.py": "...new content...", "src/b.py": "..."}}. '
    "Only include files you change. Output ONLY the JSON object."
)


@dataclasses.dataclass
class MultiFileProposal:
    files: List["FileChange"]
    diff: str  # combined multi-file unified diff
    usage: Optional[Usage] = None

    @property
    def changed(self) -> bool:
        return any(f.changed for f in self.files)

    def to_dict(self) -> dict:
        return {
            "files": [f.to_dict() for f in self.files],
            "diff": self.diff,
            "changed": self.changed,
            "usage": self.usage.to_dict() if self.usage else None,
        }


@dataclasses.dataclass
class FileChange:
    path: str
    original: str
    new_content: str

    @property
    def changed(self) -> bool:
        return self.original != self.new_content

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "new_content": self.new_content,
            "changed": self.changed,
        }


def _build_multifile_prompt(files: Dict[str, str], instruction: str) -> str:
    parts = [f"<instruction>\n{instruction}\n</instruction>"]
    for path, content in files.items():
        parts.append(f'<file path="{path}">\n{content}\n</file>')
    parts.append("Return a JSON object mapping changed paths to new content.")
    return "\n".join(parts)


def _parse_multifile_response(text: str) -> Dict[str, str]:
    """Extract ``{path: content}`` from a model JSON response (best-effort)."""
    text = _strip_fences(text).strip()
    # Find the outermost JSON object.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    files = data.get("files", data)
    if not isinstance(files, dict):
        return {}
    return {str(k): str(v) for k, v in files.items()}


def propose_multifile_edit(
    backend: LLMBackend,
    workspace: Workspace,
    *,
    paths: List[str],
    instruction: str,
    max_tokens: int = 8192,
    model: Optional[str] = None,
) -> MultiFileProposal:
    """Propose edits across multiple files; returns a combined reviewable diff."""
    sources: Dict[str, str] = {}
    for path in paths:
        sources[path] = workspace.read(path) if workspace.exists(path) else ""
    prompt = _build_multifile_prompt(sources, instruction)
    request = CompletionRequest(
        system=_MULTIFILE_SYSTEM,
        messages=[Message(role="user", content=prompt)],
        max_tokens=max_tokens,
        model=model,
    )
    raw = collect(backend.complete(request))
    new_map = _parse_multifile_response(raw)

    changes: List[FileChange] = []
    triples: List["tuple[str, str, str]"] = []
    for path in paths:
        original = sources.get(path, "")
        new_content = new_map.get(path, original)
        changes.append(FileChange(path=path, original=original, new_content=new_content))
        triples.append((path, original, new_content))
    # Include any *new* files the model returned that weren't in `paths`.
    for path, new_content in new_map.items():
        if path not in sources:
            changes.append(FileChange(path=path, original="", new_content=new_content))
            triples.append((path, "", new_content))

    diff = make_multifile_diff(triples)
    usage = _account(backend, model, _MULTIFILE_SYSTEM + prompt, raw)
    return MultiFileProposal(files=changes, diff=diff, usage=usage)


@dataclasses.dataclass
class MultiFileApplyResult:
    applied: List[str]
    reverse_diff: str

    def to_dict(self) -> dict:
        return {"applied": self.applied, "reverse_diff": self.reverse_diff}


def apply_multifile_edit(
    workspace: Workspace,
    *,
    diff: str,
    expected: Optional[Dict[str, str]] = None,
) -> MultiFileApplyResult:
    """Apply a multi-file diff to the workspace atomically.

    Computes all new contents first (raising :class:`DiffConflict` on any stale
    hunk) before writing anything, so a partial multi-file apply never leaves
    the workspace half-edited.
    """
    patches = parse_unified_diff(diff)
    sources: Dict[str, str] = {}
    for patch in patches:
        path = patch.target_path()
        if expected is not None and path in expected:
            sources[path] = expected[path]
        else:
            sources[path] = workspace.read(path) if workspace.exists(path) else ""
    try:
        new_contents = apply_multifile(sources, diff)
    except DiffConflict:
        raise
    except DiffError as exc:
        raise DiffConflict(str(exc)) from exc
    # All hunks resolved; now write.
    for path, content in new_contents.items():
        workspace.write(path, content)
    return MultiFileApplyResult(applied=list(new_contents.keys()), reverse_diff=reverse_diff(diff))
