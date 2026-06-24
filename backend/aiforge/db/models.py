"""SQLAlchemy 2.0 ORM models.

Entities: users, workspaces (one sandboxed root each), api_keys, chat_sessions,
messages, edit_history (applied diffs, reversible), and rag_index metadata.

Kept Python 3.9 compatible: ``Mapped[...]`` annotations use ``Optional`` /
``typing`` generics rather than PEP 604 ``X | None`` runtime unions.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all models."""


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    workspaces: Mapped[List["Workspace"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )
    api_keys: Mapped[List["ApiKey"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Workspace(Base):
    __tablename__ = "workspaces"
    __table_args__ = (UniqueConstraint("owner_id", "slug", name="uq_owner_slug"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    owner_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(128))
    slug: Mapped[str] = mapped_column(String(128), index=True)
    # Filesystem directory name under the data dir (isolated per workspace).
    root_dir: Mapped[str] = mapped_column(String(64), unique=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    owner: Mapped["User"] = relationship(back_populates="workspaces")
    chat_sessions: Mapped[List["ChatSession"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan"
    )
    edits: Mapped[List["EditHistory"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan"
    )
    rag_index: Mapped[Optional["RagIndexMeta"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan", uselist=False
    )


class ApiKey(Base):
    """A per-workspace API key (hashed at rest) for programmatic access."""

    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    workspace_id: Mapped[Optional[str]] = mapped_column(
        String(32), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(128))
    prefix: Mapped[str] = mapped_column(String(16), index=True)  # shown to user
    key_hash: Mapped[str] = mapped_column(String(128), index=True)  # sha256 hex
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped["User"] = relationship(back_populates="api_keys")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(256), default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    workspace: Mapped["Workspace"] = relationship(back_populates="chat_sessions")
    messages: Mapped[List["Message"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    # token / cost accounting
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(default=0.0)
    provider: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    session: Mapped["ChatSession"] = relationship(back_populates="messages")


class EditHistory(Base):
    """A record of an applied agentic edit, reversible via ``reverse_diff``."""

    __tablename__ = "edit_history"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    path: Mapped[str] = mapped_column(String(1024))
    instruction: Mapped[str] = mapped_column(Text, default="")
    forward_diff: Mapped[str] = mapped_column(Text)
    reverse_diff: Mapped[str] = mapped_column(Text)
    applied: Mapped[bool] = mapped_column(Boolean, default=True)  # False once undone
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    workspace: Mapped["Workspace"] = relationship(back_populates="edits")


class RagIndexMeta(Base):
    """Persisted metadata about a workspace's RAG index."""

    __tablename__ = "rag_index_meta"

    workspace_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True
    )
    file_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    embedder: Mapped[str] = mapped_column(String(64), default="hashing")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    workspace: Mapped["Workspace"] = relationship(back_populates="rag_index")
