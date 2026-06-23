"""Application configuration, driven by environment variables.

Uses ``pydantic-settings`` when available and falls back to a plain
``os.environ`` reader otherwise, so the package imports and the test suite runs
even in a minimal environment (``requirements-min.txt``).

All settings are prefixed ``AIFORGE_`` as environment variables, e.g.
``AIFORGE_BACKEND=anthropic``. The two vendor keys (``ANTHROPIC_API_KEY``,
``HF_TOKEN``) keep their conventional names and are read by the backends
directly.
"""
from __future__ import annotations

import os
from pathlib import Path

# Per-feature model defaults. Heavy agentic edits get Opus; chat gets Sonnet;
# latency-sensitive inline completion gets Haiku.
_DEFAULT_MODELS = {
    "complete": "claude-haiku-4-5",
    "chat": "claude-sonnet-4-6",
    "edit": "claude-opus-4-8",
}

try:  # Prefer pydantic-settings for validation when it is installed.
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        """Validated settings loaded from the environment / ``.env``."""

        model_config = SettingsConfigDict(
            env_prefix="AIFORGE_", env_file=".env", extra="ignore"
        )

        backend: str = "mock"
        workspace_root: str = "./workspace"

        # Per-feature model overrides (empty string => use backend default).
        model_complete: str = _DEFAULT_MODELS["complete"]
        model_chat: str = _DEFAULT_MODELS["chat"]
        model_edit: str = _DEFAULT_MODELS["edit"]

        # RAG settings.
        rag_chunk_lines: int = 60
        rag_chunk_overlap: int = 10
        rag_top_k: int = 6
        rag_embed_dim: int = 512
        rag_use_sentence_transformers: bool = False

        # CORS / server.
        cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
        frontend_dist: str = ""

        def model_for(self, feature: str) -> str:
            return {
                "complete": self.model_complete,
                "chat": self.model_chat,
                "edit": self.model_edit,
            }.get(feature, self.model_edit)

        def cors_origin_list(self) -> list[str]:
            return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

        def resolved_workspace_root(self) -> Path:
            return Path(self.workspace_root).expanduser().resolve()

except ImportError:  # pragma: no cover - exercised only without pydantic
    class Settings:  # type: ignore[no-redef]
        """Plain-environment fallback with the same public surface."""

        def __init__(self) -> None:
            env = os.environ.get
            self.backend = env("AIFORGE_BACKEND", "mock")
            self.workspace_root = env("AIFORGE_WORKSPACE_ROOT", "./workspace")
            self.model_complete = env("AIFORGE_MODEL_COMPLETE", _DEFAULT_MODELS["complete"])
            self.model_chat = env("AIFORGE_MODEL_CHAT", _DEFAULT_MODELS["chat"])
            self.model_edit = env("AIFORGE_MODEL_EDIT", _DEFAULT_MODELS["edit"])
            self.rag_chunk_lines = int(env("AIFORGE_RAG_CHUNK_LINES", "60"))
            self.rag_chunk_overlap = int(env("AIFORGE_RAG_CHUNK_OVERLAP", "10"))
            self.rag_top_k = int(env("AIFORGE_RAG_TOP_K", "6"))
            self.rag_embed_dim = int(env("AIFORGE_RAG_EMBED_DIM", "512"))
            self.rag_use_sentence_transformers = (
                env("AIFORGE_RAG_USE_SENTENCE_TRANSFORMERS", "false").lower() == "true"
            )
            self.cors_origins = env(
                "AIFORGE_CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173",
            )
            self.frontend_dist = env("AIFORGE_FRONTEND_DIST", "")

        def model_for(self, feature: str) -> str:
            return {
                "complete": self.model_complete,
                "chat": self.model_chat,
                "edit": self.model_edit,
            }.get(feature, self.model_edit)

        def cors_origin_list(self) -> list[str]:
            return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

        def resolved_workspace_root(self) -> Path:
            return Path(self.workspace_root).expanduser().resolve()


_settings: "Settings | None" = None


def get_settings() -> "Settings":
    """Return a process-wide cached :class:`Settings` instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
