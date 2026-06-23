"""A robust unified-diff parser and applier, in pure Python.

This is the engine behind the agentic-edit "apply" flow. It parses standard
unified diffs (``--- a/x``/``+++ b/x`` headers, ``@@ -l,s +l,s @@`` hunks,
context/add/remove lines) and applies them to in-memory file contents. It is
deliberately strict about context so a malformed or stale diff fails loudly
rather than corrupting a file.

Design notes:

- Hunk application is line-oriented and verifies context/removed lines against
  the source. If the source has drifted, we attempt a bounded fuzzy re-locate
  (search a small window around the hunk's declared position) before failing.
- :func:`reverse_diff` produces the inverse diff, which makes round-trip
  testing (apply then un-apply) straightforward and is genuinely useful for an
  "undo" feature.
"""
from __future__ import annotations

import dataclasses
import re
from typing import List, Optional, Tuple

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class DiffError(Exception):
    """Raised for malformed diffs or hunks that do not apply cleanly."""


@dataclasses.dataclass
class Hunk:
    old_start: int  # 1-based
    old_count: int
    new_start: int  # 1-based
    new_count: int
    lines: List[str]  # each begins with ' ', '+', '-', or '\\'

    def header(self) -> str:
        return f"@@ -{self.old_start},{self.old_count} +{self.new_start},{self.new_count} @@"


@dataclasses.dataclass
class FilePatch:
    old_path: str
    new_path: str
    hunks: List[Hunk]

    @property
    def is_creation(self) -> bool:
        return self.old_path == "/dev/null"

    @property
    def is_deletion(self) -> bool:
        return self.new_path == "/dev/null"

    def target_path(self) -> str:
        path = self.new_path if not self.is_deletion else self.old_path
        return _strip_prefix(path)


def _strip_prefix(path: str) -> str:
    """Strip a leading ``a/`` or ``b/`` from a diff path."""
    if path in ("/dev/null", ""):
        return path
    for prefix in ("a/", "b/"):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def parse_unified_diff(text: str) -> List[FilePatch]:
    """Parse a (possibly multi-file) unified diff into :class:`FilePatch` objects."""
    lines = text.splitlines()
    patches: List[FilePatch] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        # Skip ``diff --git`` and other git metadata lines until a file header.
        if line.startswith("--- "):
            old_path = line[4:].strip().split("\t")[0]
            if i + 1 >= n or not lines[i + 1].startswith("+++ "):
                raise DiffError("'---' header not followed by '+++' header")
            new_path = lines[i + 1][4:].strip().split("\t")[0]
            i += 2
            hunks, i = _parse_hunks(lines, i)
            patches.append(FilePatch(old_path=old_path, new_path=new_path, hunks=hunks))
            continue
        i += 1
    if not patches:
        raise DiffError("no file headers ('--- '/'+++ ') found in diff")
    return patches


def _parse_hunks(lines: List[str], i: int) -> Tuple[List[Hunk], int]:
    hunks: List[Hunk] = []
    n = len(lines)
    while i < n:
        m = _HUNK_RE.match(lines[i])
        if not m:
            break
        old_start = int(m.group(1))
        old_count = int(m.group(2)) if m.group(2) is not None else 1
        new_start = int(m.group(3))
        new_count = int(m.group(4)) if m.group(4) is not None else 1
        i += 1
        body: List[str] = []
        seen_old = seen_new = 0
        while i < n:
            ln = lines[i]
            if ln.startswith("--- ") or _HUNK_RE.match(ln):
                break
            if ln.startswith(("diff ", "index ")):
                break
            if not ln:
                # A bare empty line is a context line for an empty source line.
                body.append(" ")
                seen_old += 1
                seen_new += 1
                i += 1
            elif ln[0] == " ":
                body.append(ln)
                seen_old += 1
                seen_new += 1
                i += 1
            elif ln[0] == "+":
                body.append(ln)
                seen_new += 1
                i += 1
            elif ln[0] == "-":
                body.append(ln)
                seen_old += 1
                i += 1
            elif ln[0] == "\\":  # "\ No newline at end of file"
                body.append(ln)
                i += 1
            else:
                break
            if seen_old >= old_count and seen_new >= new_count:
                # Hunk satisfied; stop unless the next line clearly belongs.
                if i < n and lines[i][:1] in (" ", "+", "-"):
                    nxt = lines[i]
                    if nxt and nxt[0] == " ":
                        # trailing context beyond the declared count; allow it.
                        pass
                    else:
                        break
                else:
                    break
        hunks.append(
            Hunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                lines=body,
            )
        )
    return hunks, i


# --------------------------------------------------------------------------
# Applying
# --------------------------------------------------------------------------
def apply_patch(original: str, patch: FilePatch) -> str:
    """Apply a single :class:`FilePatch` to ``original`` text, returning new text."""
    if patch.is_creation:
        return _build_created_file(patch)

    original_lines = original.splitlines()
    # Apply hunks from the bottom up so earlier edits don't shift later offsets.
    result = list(original_lines)
    for hunk in sorted(patch.hunks, key=lambda h: h.old_start, reverse=True):
        result = _apply_hunk(result, hunk)
    new_text = "\n".join(result)
    # Preserve a trailing newline if the original had one and the diff did not
    # explicitly remove it.
    if original.endswith("\n") and not new_text.endswith("\n"):
        new_text += "\n"
    return new_text


def _build_created_file(patch: FilePatch) -> str:
    out: List[str] = []
    for hunk in patch.hunks:
        for ln in hunk.lines:
            if ln.startswith("+"):
                out.append(ln[1:])
    text = "\n".join(out)
    return text + "\n" if text else text


def _hunk_old_lines(hunk: Hunk) -> List[str]:
    """The lines the hunk expects to find in the source (context + removed)."""
    return [ln[1:] for ln in hunk.lines if ln[:1] in (" ", "-")]


def _hunk_new_lines(hunk: Hunk) -> List[str]:
    """The lines the hunk produces (context + added)."""
    return [ln[1:] for ln in hunk.lines if ln[:1] in (" ", "+")]


def _apply_hunk(lines: List[str], hunk: Hunk) -> List[str]:
    old_block = _hunk_old_lines(hunk)
    new_block = _hunk_new_lines(hunk)
    # Preferred position: declared old_start (converted to 0-based).
    pos = hunk.old_start - 1
    if not _matches(lines, pos, old_block):
        pos = _relocate(lines, pos, old_block)
        if pos is None:
            raise DiffError(
                f"hunk does not apply; context mismatch near line {hunk.old_start}"
            )
    return lines[:pos] + new_block + lines[pos + len(old_block):]


def _matches(lines: List[str], pos: int, block: List[str]) -> bool:
    if pos < 0 or pos + len(block) > len(lines):
        return False
    return lines[pos:pos + len(block)] == block


def _relocate(lines: List[str], guess: int, block: List[str], window: int = 50) -> Optional[int]:
    """Search a bounded window around ``guess`` for an exact ``block`` match."""
    if not block:
        return guess if 0 <= guess <= len(lines) else None
    lo = max(0, guess - window)
    hi = min(len(lines) - len(block), guess + window)
    for offset in range(0, max(hi - lo, 0) + 1):
        # Probe outward from the guess for the closest match.
        for cand in {guess - offset, guess + offset}:
            if lo <= cand <= hi and _matches(lines, cand, block):
                return cand
    return None


def apply_unified_diff(original: str, diff_text: str) -> str:
    """Parse ``diff_text`` and apply its first file patch to ``original``."""
    patches = parse_unified_diff(diff_text)
    return apply_patch(original, patches[0])


# --------------------------------------------------------------------------
# Reversing
# --------------------------------------------------------------------------
def reverse_diff(diff_text: str) -> str:
    """Return the inverse of a unified diff (apply it to undo the original)."""
    patches = parse_unified_diff(diff_text)
    out: List[str] = []
    for patch in patches:
        out.append(f"--- {patch.new_path}")
        out.append(f"+++ {patch.old_path}")
        for hunk in patch.hunks:
            rev_lines: List[str] = []
            for ln in hunk.lines:
                tag, rest = ln[:1], ln[1:]
                if tag == "+":
                    rev_lines.append("-" + rest)
                elif tag == "-":
                    rev_lines.append("+" + rest)
                else:
                    rev_lines.append(ln)
            out.append(
                f"@@ -{hunk.new_start},{hunk.new_count} "
                f"+{hunk.old_start},{hunk.old_count} @@"
            )
            out.extend(rev_lines)
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# Generating
# --------------------------------------------------------------------------
def make_unified_diff(path: str, old: str, new: str, *, context: int = 3) -> str:
    """Produce a unified diff from ``old`` to ``new`` for ``path``.

    Thin wrapper over :func:`difflib.unified_diff` that yields output our parser
    round-trips. Used to turn a model's "full new content" into a reviewable
    diff for the UI.
    """
    import difflib

    old_lines = old.splitlines(keepends=False)
    new_lines = new.splitlines(keepends=False)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=context,
        lineterm="",
    )
    return "\n".join(diff) + "\n"
