"""Sandboxed workspace file operations.

Every path that crosses the API is untrusted. :class:`Workspace` confines all
reads and writes to a single root directory and rejects any attempt to escape
it via ``..``, absolute paths, or symlinks. The resolution strategy:

1. Reject absolute inputs and inputs containing a ``..`` component up front.
2. Join to the root and resolve to a canonical path.
3. Verify the canonical path is still inside the (canonical) root.

This catches ``..`` traversal, symlink escapes (the resolved real path leaves
the root), and absolute-path injection.
"""
from __future__ import annotations

import dataclasses
import os
import shutil
from pathlib import Path
from typing import List, Optional

# Directories never walked for the tree / RAG index.
_IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".aiforge",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
}
_IGNORE_SUFFIXES = {".pyc", ".pyo", ".so", ".o", ".class", ".lock"}


class WorkspaceError(Exception):
    """Base class for workspace errors."""


class PathTraversalError(WorkspaceError):
    """Raised when a requested path would escape the workspace root."""


class NotFoundError(WorkspaceError):
    """Raised when a path does not exist."""


@dataclasses.dataclass
class TreeNode:
    """A node in the workspace file tree."""

    name: str
    path: str  # POSIX-style path relative to the workspace root
    type: str  # "file" | "dir"
    children: Optional[List["TreeNode"]] = None

    def to_dict(self) -> dict:
        node = {"name": self.name, "path": self.path, "type": self.type}
        if self.children is not None:
            node["children"] = [c.to_dict() for c in self.children]
        return node


class Workspace:
    """A sandboxed view of a single root directory."""

    def __init__(self, root: os.PathLike | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # -- path safety --------------------------------------------------------
    def resolve(self, rel_path: str) -> Path:
        """Resolve ``rel_path`` against the root, enforcing the sandbox.

        Raises :class:`PathTraversalError` if the resolved path escapes the
        root. The path need not exist (so this is usable for create/write).
        """
        if rel_path is None:
            raise PathTraversalError("path is required")
        # Normalise separators and strip a leading slash so "/foo" is treated
        # as workspace-relative, not filesystem-absolute.
        cleaned = str(rel_path).replace("\\", "/").lstrip("/")
        if cleaned in ("", "."):
            return self.root
        candidate = Path(cleaned)
        if candidate.is_absolute():
            raise PathTraversalError(f"absolute paths are not allowed: {rel_path!r}")
        if ".." in candidate.parts:
            raise PathTraversalError(f"path traversal is not allowed: {rel_path!r}")
        full = (self.root / candidate).resolve()
        # Final defence: the resolved real path must remain within the root.
        # Covers symlinks whose target is outside the sandbox.
        if not self._is_within_root(full):
            raise PathTraversalError(f"path escapes workspace: {rel_path!r}")
        return full

    def _is_within_root(self, full: Path) -> bool:
        try:
            full.relative_to(self.root)
            return True
        except ValueError:
            return False

    def relativize(self, full: Path) -> str:
        return full.resolve().relative_to(self.root).as_posix()

    # -- tree ---------------------------------------------------------------
    def tree(self, rel_path: str = "") -> TreeNode:
        """Return the file tree rooted at ``rel_path`` (default: workspace root)."""
        base = self.resolve(rel_path)
        if not base.exists():
            raise NotFoundError(rel_path)
        return self._build_node(base)

    def _build_node(self, path: Path) -> TreeNode:
        rel = "" if path == self.root else self.relativize(path)
        name = self.root.name if path == self.root else path.name
        if path.is_dir():
            children: List[TreeNode] = []
            try:
                entries = sorted(
                    path.iterdir(),
                    key=lambda p: (p.is_file(), p.name.lower()),
                )
            except OSError:
                entries = []
            for entry in entries:
                if entry.is_dir() and entry.name in _IGNORE_DIRS:
                    continue
                if entry.is_file() and entry.suffix in _IGNORE_SUFFIXES:
                    continue
                if entry.is_symlink() and not self._is_within_root(entry.resolve()):
                    continue
                children.append(self._build_node(entry))
            return TreeNode(name=name, path=rel, type="dir", children=children)
        return TreeNode(name=name, path=rel, type="file")

    def list_files(self, rel_path: str = "") -> List[str]:
        """Return a flat list of file paths (POSIX, relative) under ``rel_path``."""
        base = self.resolve(rel_path)
        results: List[str] = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS]
            for fname in filenames:
                if Path(fname).suffix in _IGNORE_SUFFIXES:
                    continue
                full = Path(dirpath) / fname
                if full.is_symlink() and not self._is_within_root(full.resolve()):
                    continue
                results.append(self.relativize(full))
        return sorted(results)

    # -- file CRUD ----------------------------------------------------------
    def read(self, rel_path: str) -> str:
        full = self.resolve(rel_path)
        if not full.exists() or not full.is_file():
            raise NotFoundError(rel_path)
        return full.read_text(encoding="utf-8")

    def write(self, rel_path: str, content: str) -> None:
        full = self.resolve(rel_path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")

    def create(self, rel_path: str, content: str = "") -> None:
        full = self.resolve(rel_path)
        if full.exists():
            raise WorkspaceError(f"already exists: {rel_path!r}")
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")

    def delete(self, rel_path: str) -> None:
        full = self.resolve(rel_path)
        if not full.exists():
            raise NotFoundError(rel_path)
        if full == self.root:
            raise WorkspaceError("refusing to delete the workspace root")
        if full.is_dir():
            shutil.rmtree(full)
        else:
            full.unlink()

    def exists(self, rel_path: str) -> bool:
        try:
            return self.resolve(rel_path).exists()
        except PathTraversalError:
            return False
