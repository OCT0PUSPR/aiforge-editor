"""Authentication endpoints: register, login, refresh, me."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...config import Settings
from ...db import User, Workspace, get_db
from ...security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from ...security.tokens import TokenError
from ...workspace import slugify
from ..deps import get_current_user, settings_dep
from ..schemas import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _issue_tokens(user_id: str, ttl: int) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user_id),
        refresh_token=create_refresh_token(user_id),
        expires_in=ttl,
    )


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(
    body: RegisterRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
) -> TokenResponse:
    if not settings.allow_registration:
        raise HTTPException(status_code=403, detail="registration is disabled")
    existing = db.execute(
        select(User).where((User.email == body.email) | (User.username == body.username))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail="email or username already taken")
    user = User(
        email=body.email,
        username=body.username,
        password_hash=hash_password(body.password),
    )
    db.add(user)
    db.flush()
    # Every new user gets a default workspace.
    ws = Workspace(owner_id=user.id, name="Default", slug=slugify("default"))
    db.add(ws)
    db.commit()
    return _issue_tokens(user.id, settings.access_token_ttl_seconds)


@router.post("/login", response_model=TokenResponse)
def login(
    body: LoginRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
) -> TokenResponse:
    user = db.execute(
        select(User).where((User.username == body.username) | (User.email == body.username))
    ).scalar_one_or_none()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="account disabled")
    return _issue_tokens(user.id, settings.access_token_ttl_seconds)


@router.post("/refresh", response_model=TokenResponse)
def refresh(
    body: RefreshRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
) -> TokenResponse:
    try:
        payload = decode_token(body.refresh_token, expected_type="refresh")
    except TokenError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    user = db.get(User, payload.get("sub"))
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="user not found")
    return _issue_tokens(user.id, settings.access_token_ttl_seconds)


@router.get("/me", response_model=UserResponse)
def me(user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse(
        id=user.id, email=user.email, username=user.username, is_admin=user.is_admin
    )
