"""Database engine / session management.

SQLite by default; Postgres (or any SQLAlchemy URL) via ``AIFORGE_DATABASE_URL``.
Provides an engine factory, a session factory, a FastAPI dependency, and a
``init_db`` that creates tables (used in dev/tests; Alembic owns production
migrations).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import get_settings
from .models import Base

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def _make_engine(url: str) -> Engine:
    connect_args = {}
    if url.startswith("sqlite"):
        # Needed for SQLite when used across threads (FastAPI runs handlers in
        # a threadpool). check_same_thread=False is safe with scoped sessions.
        connect_args = {"check_same_thread": False}
    return create_engine(url, future=True, connect_args=connect_args, pool_pre_ping=True)


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        settings = get_settings()
        # Ensure the data dir exists for the default sqlite path.
        settings.resolved_data_dir()
        _engine = _make_engine(settings.database_url)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    return _engine


def session_factory() -> sessionmaker:
    if _SessionLocal is None:
        get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


def init_db() -> None:
    """Create all tables (dev/test convenience; prod uses Alembic)."""
    engine = get_engine()
    Base.metadata.create_all(engine)


def reset_engine() -> None:
    """Drop the cached engine/session factory (used by tests)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on error."""
    factory = session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a session (no auto-commit; handlers commit)."""
    factory = session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()
