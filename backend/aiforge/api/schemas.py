"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


# -- auth -------------------------------------------------------------------
class RegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=256)


class LoginRequest(BaseModel):
    username: str
    password: str = Field(max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    is_admin: bool


# -- workspaces -------------------------------------------------------------
class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)


class WorkspaceResponse(BaseModel):
    id: str
    name: str
    slug: str


class WorkspaceUsage(BaseModel):
    files: int
    bytes: int
    max_files: int
    max_bytes: int


# -- api keys ---------------------------------------------------------------
class ApiKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    workspace_id: Optional[str] = None


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    prefix: str
    key: Optional[str] = None  # full key returned only on creation


# -- files ------------------------------------------------------------------
class FileSave(BaseModel):
    path: str = Field(max_length=1024)
    content: str


class FileCreate(BaseModel):
    path: str = Field(max_length=1024)
    content: str = ""


class FileRename(BaseModel):
    src: str = Field(max_length=1024)
    dst: str = Field(max_length=1024)


# -- AI ---------------------------------------------------------------------
class CompleteRequest(BaseModel):
    prefix: str = Field(max_length=200_000)
    suffix: str = Field(default="", max_length=200_000)
    language: str = ""
    path: str = ""
    max_tokens: int = Field(default=256, ge=1, le=2048)


class ChatTurn(BaseModel):
    role: str
    content: str = Field(max_length=200_000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20_000)
    open_path: str = ""
    open_content: str = Field(default="", max_length=400_000)
    history: List[ChatTurn] = Field(default_factory=list, max_length=50)
    top_k: int = Field(default=6, ge=1, le=20)
    session_id: Optional[str] = None


class EditRequest(BaseModel):
    path: str = Field(max_length=1024)
    instruction: str = Field(min_length=1, max_length=20_000)


class MultiEditRequest(BaseModel):
    paths: List[str] = Field(min_length=1, max_length=20)
    instruction: str = Field(min_length=1, max_length=20_000)


class EditApplyRequest(BaseModel):
    path: Optional[str] = None
    diff: Optional[str] = None
    new_content: Optional[str] = None
    expected_original: Optional[str] = None
    multifile: bool = False
    instruction: str = ""
