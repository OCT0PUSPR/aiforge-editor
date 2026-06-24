"""Shared application services container.

Holds singletons (settings, workspace manager, rate limiter) and per-workspace
RAG indexers (lazily created, cached, and persisted). Attached to
``app.state.services`` and reachable from request handlers.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, Optional

from ..config import Settings
from ..llm import build_backend
from ..llm.base import LLMBackend
from ..rag.indexer import RagIndexer
from ..security.ratelimit import make_limiter
from ..workspace import Workspace, WorkspaceManager


class Services:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.manager = WorkspaceManager(settings)
        self.limiter = make_limiter(settings.redis_url or None)
        self._indexers: Dict[str, RagIndexer] = {}
        self._lock = threading.Lock()

    # -- LLM ----------------------------------------------------------------
    def backend_for(self, feature: str) -> LLMBackend:
        # Inline completion can be routed to our own local model.
        if feature == "complete" and self.settings.complete_backend:
            from ..llm import get_backend

            return get_backend(
                self.settings.complete_backend,
                model=self.settings.model_for(feature),
                local_model_dir=self.settings.local_model_dir or None,
            )
        return build_backend(self.settings, model=self.settings.model_for(feature))

    # -- RAG indexers (per workspace) ---------------------------------------
    def _persist_dir(self, root_dir: str) -> Optional[str]:
        if not self.settings.rag_persist:
            return None
        return str(self.manager.root_for(root_dir).parent.parent / "rag" / root_dir)

    def indexer_for(self, root_dir: str) -> RagIndexer:
        with self._lock:
            idx = self._indexers.get(root_dir)
            if idx is not None:
                return idx
            fs: Workspace = self.manager.fs(root_dir)
            persist = self._persist_dir(root_dir)
            idx = RagIndexer(
                fs,
                embed_dim=self.settings.rag_embed_dim,
                chunk_lines=self.settings.rag_chunk_lines,
                chunk_overlap=self.settings.rag_chunk_overlap,
                embedder=self.settings.rag_embedder,
                st_model=self.settings.rag_st_model,
                persist_dir=persist,
            )
            if persist and Path(persist).exists():
                idx.load(persist)
            self._indexers[root_dir] = idx
            return idx

    def reset_indexer(self, root_dir: str) -> None:
        with self._lock:
            self._indexers.pop(root_dir, None)
