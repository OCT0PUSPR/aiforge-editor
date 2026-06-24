"""Workspace and API-key management endpoints."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...db import ApiKey, User, Workspace, get_db
from ...security import generate_api_key
from ...workspace import slugify
from ..deps import get_current_user, get_services, get_workspace
from ..schemas import (
    ApiKeyCreate,
    ApiKeyResponse,
    WorkspaceCreate,
    WorkspaceResponse,
    WorkspaceUsage,
)
from ..services import Services

router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])
keys_router = APIRouter(prefix="/api/keys", tags=["api-keys"])


@router.get("", response_model=List[WorkspaceResponse])
def list_workspaces(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> List[WorkspaceResponse]:
    rows = (
        db.execute(
            select(Workspace).where(Workspace.owner_id == user.id).order_by(Workspace.created_at)
        )
        .scalars()
        .all()
    )
    return [WorkspaceResponse(id=w.id, name=w.name, slug=w.slug) for w in rows]


@router.post("", response_model=WorkspaceResponse, status_code=201)
def create_workspace(
    body: WorkspaceCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    services: Services = Depends(get_services),
) -> WorkspaceResponse:
    slug = slugify(body.name)
    # Ensure unique slug per owner.
    exists = db.execute(
        select(Workspace).where(Workspace.owner_id == user.id, Workspace.slug == slug)
    ).scalar_one_or_none()
    if exists is not None:
        raise HTTPException(status_code=409, detail="workspace name already in use")
    ws = Workspace(owner_id=user.id, name=body.name, slug=slug)
    db.add(ws)
    db.commit()
    services.manager.provision(ws.root_dir)
    return WorkspaceResponse(id=ws.id, name=ws.name, slug=ws.slug)


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
def get_workspace_detail(ws: Workspace = Depends(get_workspace)) -> WorkspaceResponse:
    return WorkspaceResponse(id=ws.id, name=ws.name, slug=ws.slug)


@router.get("/{workspace_id}/usage", response_model=WorkspaceUsage)
def workspace_usage(
    ws: Workspace = Depends(get_workspace),
    services: Services = Depends(get_services),
) -> WorkspaceUsage:
    fs = services.manager.fs(ws.root_dir)
    usage = fs.usage()
    return WorkspaceUsage(
        files=usage["files"],
        bytes=usage["bytes"],
        max_files=services.settings.max_workspace_files,
        max_bytes=services.settings.max_workspace_bytes,
    )


@router.delete("/{workspace_id}", status_code=204)
def delete_workspace(
    ws: Workspace = Depends(get_workspace),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    # Don't allow deleting the last remaining workspace.
    count = db.execute(select(Workspace).where(Workspace.owner_id == user.id)).scalars().all()
    if len(count) <= 1:
        raise HTTPException(status_code=400, detail="cannot delete the last workspace")
    db.delete(ws)
    db.commit()


# -- API keys ---------------------------------------------------------------
@keys_router.get("", response_model=List[ApiKeyResponse])
def list_keys(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> List[ApiKeyResponse]:
    rows = (
        db.execute(select(ApiKey).where(ApiKey.user_id == user.id, ApiKey.is_active.is_(True)))
        .scalars()
        .all()
    )
    return [ApiKeyResponse(id=k.id, name=k.name, prefix=k.prefix) for k in rows]


@keys_router.post("", response_model=ApiKeyResponse, status_code=201)
def create_key(
    body: ApiKeyCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApiKeyResponse:
    if body.workspace_id is not None:
        ws = db.get(Workspace, body.workspace_id)
        if ws is None or ws.owner_id != user.id:
            raise HTTPException(status_code=404, detail="workspace not found")
    full, prefix, key_hash = generate_api_key()
    key = ApiKey(
        user_id=user.id,
        workspace_id=body.workspace_id,
        name=body.name,
        prefix=prefix,
        key_hash=key_hash,
    )
    db.add(key)
    db.commit()
    # Full key is returned exactly once.
    return ApiKeyResponse(id=key.id, name=key.name, prefix=key.prefix, key=full)


@keys_router.delete("/{key_id}", status_code=204)
def revoke_key(
    key_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    key = db.get(ApiKey, key_id)
    if key is None or key.user_id != user.id:
        raise HTTPException(status_code=404, detail="key not found")
    key.is_active = False
    db.commit()
