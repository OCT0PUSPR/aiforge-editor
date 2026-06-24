"""Maps DB :class:`Workspace` rows to isolated filesystem roots.

Each workspace gets its own directory under ``<data_dir>/workspaces/<root_dir>``.
The manager hands out :class:`Workspace` (FS-sandbox) objects scoped to those
roots, applying per-workspace quotas from settings. This is what enforces
multi-user isolation: one user's workspace can never resolve a path into
another's, because each FS sandbox has a distinct, separate root.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..config import Settings
from .files import Quota
from .files import Workspace as FsWorkspace

_SLUG_RE = re.compile(r"[^a-z0-9-]+")


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    return slug or "workspace"


class WorkspaceManager:
    """Resolves and provisions per-workspace filesystem sandboxes."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._base = settings.resolved_data_dir() / "workspaces"
        self._base.mkdir(parents=True, exist_ok=True)

    def _quota(self) -> Quota:
        return Quota(
            max_file_bytes=self.settings.max_file_bytes,
            max_workspace_files=self.settings.max_workspace_files,
            max_workspace_bytes=self.settings.max_workspace_bytes,
        )

    def root_for(self, root_dir: str) -> Path:
        # root_dir is a server-generated hex id; still guard against traversal.
        safe = _SLUG_RE.sub("", root_dir.lower())
        if not safe:
            raise ValueError("invalid workspace root id")
        return (self._base / safe).resolve()

    def fs(self, root_dir: str) -> FsWorkspace:
        """Return a sandboxed :class:`Workspace` for the given root id."""
        root = self.root_for(root_dir)
        return FsWorkspace(root, quota=self._quota())

    def provision(self, root_dir: str) -> Path:
        """Create the directory for a new workspace and return its path."""
        root = self.root_for(root_dir)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def seed(self, root_dir: str, files: "dict[str, str]") -> None:
        """Seed a workspace with initial files (used for demo/default ws)."""
        fs = self.fs(root_dir)
        for rel, content in files.items():
            if not fs.exists(rel):
                fs.write(rel, content)
