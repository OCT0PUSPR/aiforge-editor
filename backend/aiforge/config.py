"""Application configuration, driven by environment variables.

Uses ``pydantic-settings``. All app settings are prefixed ``AIFORGE_`` as
environment variables, e.g. ``AIFORGE_BACKEND=anthropic``. Vendor keys keep
their conventional names (``ANTHROPIC_API_KEY``, ``HF_TOKEN``) and are read by
the LLM backends directly.

Everything has a safe default so the app boots offline with no secrets.
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

# Per-feature model defaults. Heavy agentic edits get Opus; chat gets Sonnet;
# latency-sensitive inline completion gets Haiku.
_DEFAULT_MODELS = {
    "complete": "claude-haiku-4-5",
    "chat": "claude-sonnet-4-6",
    "edit": "claude-opus-4-8",
}


class Settings(BaseSettings):
    """Validated settings loaded from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="AIFORGE_", env_file=".env", extra="ignore", case_sensitive=False
    )

    # -- LLM ----------------------------------------------------------------
    backend: str = "mock"
    # Failover order tried by the resilient backend (comma-separated).
    backend_failover: str = "anthropic,huggingface,mock"
    enable_failover: bool = False
    llm_timeout: float = 60.0
    llm_max_retries: int = 2

    model_complete: str = _DEFAULT_MODELS["complete"]
    model_chat: str = _DEFAULT_MODELS["chat"]
    model_edit: str = _DEFAULT_MODELS["edit"]

    # Directory of a trained local code model (powers inline completion when the
    # complete backend is routed to "local"). Empty => no local model.
    local_model_dir: str = ""
    # Per-feature backend override. Empty => use ``backend``. Set
    # ``complete_backend=local`` to serve inline completion from our own model
    # while chat/edit keep Anthropic in production.
    complete_backend: str = ""

    # -- Storage ------------------------------------------------------------
    # Parent directory under which each workspace gets its own sandboxed root.
    data_dir: str = "./data"
    workspace_root: str = "./workspace"  # default single-workspace fallback
    database_url: str = "sqlite:///./data/aiforge.db"

    # Quotas (per workspace).
    max_file_bytes: int = 2_000_000  # 2 MB per file
    max_workspace_files: int = 5000
    max_workspace_bytes: int = 200_000_000  # 200 MB per workspace
    max_request_bytes: int = 4_000_000  # request body cap

    # -- RAG ----------------------------------------------------------------
    rag_chunk_lines: int = 60
    rag_chunk_overlap: int = 10
    rag_top_k: int = 6
    rag_embed_dim: int = 512
    rag_embedder: str = "hashing"  # hashing | sentence-transformers
    rag_st_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_vector_store: str = "numpy"  # numpy | qdrant
    rag_persist: bool = True

    # -- Auth / security ----------------------------------------------------
    # If unset, a random secret is generated per-process (fine for dev; set a
    # stable value in production so tokens survive restarts).
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    access_token_ttl_seconds: int = 3600  # 1 hour
    refresh_token_ttl_seconds: int = 1_209_600  # 14 days
    allow_registration: bool = True

    # Rate limiting (token bucket, per user/IP).
    rate_limit_rpm: int = 120  # general endpoints
    rate_limit_ai_rpm: int = 30  # AI endpoints (more expensive)
    redis_url: str = ""  # optional; in-memory limiter if empty

    # -- Server / CORS ------------------------------------------------------
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    frontend_dist: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    request_timeout: float = 120.0
    log_level: str = "info"
    log_json: bool = True

    # ----------------------------------------------------------------------
    def model_for(self, feature: str) -> str:
        return {
            "complete": self.model_complete,
            "chat": self.model_chat,
            "edit": self.model_edit,
        }.get(feature, self.model_edit)

    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def failover_list(self) -> List[str]:
        return [b.strip() for b in self.backend_failover.split(",") if b.strip()]

    def resolved_data_dir(self) -> Path:
        p = Path(self.data_dir).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def resolved_workspace_root(self) -> Path:
        return Path(self.workspace_root).expanduser().resolve()

    def effective_jwt_secret(self) -> str:
        return self.jwt_secret or _process_secret()


_PROCESS_SECRET = secrets.token_urlsafe(48)


def _process_secret() -> str:
    return _PROCESS_SECRET


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cached settings (used by tests that tweak the environment)."""
    get_settings.cache_clear()
