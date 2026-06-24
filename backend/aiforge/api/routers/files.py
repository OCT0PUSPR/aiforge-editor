"""Workspace file CRUD endpoints (sandboxed per workspace)."""

from __future__ import annotations

from typing import NoReturn

from fastapi import APIRouter, Depends, HTTPException, Query

from ...workspace import (
    NotFoundError,
    PathTraversalError,
    QuotaError,
    ValidationError,
    WorkspaceError,
)
from ...workspace import (
    Workspace as FsWorkspace,
)
from ..deps import get_fs, rate_limit
from ..schemas import FileCreate, FileRename, FileSave

router = APIRouter(
    prefix="/api/workspaces/{workspace_id}/files",
    tags=["files"],
    dependencies=[Depends(rate_limit)],
)


def _handle(exc: Exception) -> NoReturn:
    if isinstance(exc, NotFoundError):
        raise HTTPException(status_code=404, detail="not found")
    if isinstance(exc, PathTraversalError):
        raise HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, (QuotaError,)):
        raise HTTPException(status_code=413, detail=str(exc))
    if isinstance(exc, ValidationError):
        raise HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, WorkspaceError):
        raise HTTPException(status_code=409, detail=str(exc))
    raise exc


@router.get("/tree")
def tree(path: str = Query(""), fs: FsWorkspace = Depends(get_fs)) -> dict:
    try:
        return fs.tree(path).to_dict()
    except Exception as exc:
        _handle(exc)


@router.get("")
def read_file(path: str = Query(...), fs: FsWorkspace = Depends(get_fs)) -> dict:
    try:
        return {"path": path, "content": fs.read(path)}
    except Exception as exc:
        _handle(exc)


@router.put("")
def save_file(body: FileSave, fs: FsWorkspace = Depends(get_fs)) -> dict:
    try:
        fs.write(body.path, body.content)
        return {"path": body.path, "saved": True}
    except Exception as exc:
        _handle(exc)


@router.post("")
def create_file(body: FileCreate, fs: FsWorkspace = Depends(get_fs)) -> dict:
    try:
        fs.create(body.path, body.content)
        return {"path": body.path, "created": True}
    except Exception as exc:
        _handle(exc)


@router.post("/rename")
def rename_file(body: FileRename, fs: FsWorkspace = Depends(get_fs)) -> dict:
    try:
        fs.rename(body.src, body.dst)
        return {"src": body.src, "dst": body.dst, "renamed": True}
    except Exception as exc:
        _handle(exc)


@router.delete("")
def delete_file(path: str = Query(...), fs: FsWorkspace = Depends(get_fs)) -> dict:
    try:
        fs.delete(path)
        return {"path": path, "deleted": True}
    except Exception as exc:
        _handle(exc)
