"""FastAPI dependencies: auth, current user/workspace, rate limiting."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, Header, HTTPException, Path, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..db import ApiKey, User, Workspace, get_db
from ..security import decode_token, hash_api_key, is_api_key
from ..security.tokens import TokenError
from ..workspace import Workspace as FsWorkspace
from .services import Services


def get_services(request: Request) -> Services:
    return request.app.state.services


def settings_dep() -> Settings:
    return get_settings()


# -- authentication ---------------------------------------------------------
def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


def get_current_user(
    authorization: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    """Resolve the authenticated user from a JWT *or* an API key bearer token."""
    token = _extract_bearer(authorization)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if is_api_key(token):
        key_hash = hash_api_key(token)
        api_key = db.execute(
            select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.is_active.is_(True))
        ).scalar_one_or_none()
        if api_key is None:
            raise HTTPException(status_code=401, detail="invalid API key")
        api_key.last_used_at = datetime.now(timezone.utc)
        db.commit()
        user = db.get(User, api_key.user_id)
    else:
        try:
            payload = decode_token(token, expected_type="access")
        except TokenError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        user = db.get(User, payload.get("sub"))

    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="user not found or inactive")
    return user


def get_current_user_optional(
    authorization: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
) -> Optional[User]:
    try:
        return get_current_user(authorization=authorization, db=db)
    except HTTPException:
        return None


# -- workspace resolution ---------------------------------------------------
def get_workspace(
    workspace_id: str = Path(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Workspace:
    """Resolve a workspace owned by the current user (404 otherwise)."""
    ws = db.get(Workspace, workspace_id)
    if ws is None or ws.owner_id != user.id:
        raise HTTPException(status_code=404, detail="workspace not found")
    return ws


def get_fs(
    ws: Workspace = Depends(get_workspace),
    services: Services = Depends(get_services),
) -> FsWorkspace:
    return services.manager.fs(ws.root_dir)


# -- rate limiting ----------------------------------------------------------
def _client_key(request: Request, user: Optional[User]) -> str:
    if user is not None:
        return f"user:{user.id}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


def rate_limit(
    request: Request,
    user: User = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> None:
    key = _client_key(request, user)
    if not services.limiter.allow(key, services.settings.rate_limit_rpm):
        raise HTTPException(status_code=429, detail="rate limit exceeded")


def ai_rate_limit(
    request: Request,
    user: User = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> None:
    key = f"ai:{_client_key(request, user)}"
    if not services.limiter.allow(key, services.settings.rate_limit_ai_rpm):
        raise HTTPException(status_code=429, detail="AI rate limit exceeded")
