"""RAG indexing + search endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ...db import RagIndexMeta, Workspace, get_db
from ...observability import metrics
from ..deps import get_services, get_workspace, rate_limit
from ..services import Services

router = APIRouter(
    prefix="/api/workspaces/{workspace_id}/rag",
    tags=["rag"],
    dependencies=[Depends(rate_limit)],
)


def _persist_meta(db: Session, workspace_id: str, stats: dict) -> None:
    meta = db.get(RagIndexMeta, workspace_id)
    if meta is None:
        meta = RagIndexMeta(workspace_id=workspace_id)
        db.add(meta)
    meta.file_count = stats.get("files", 0)
    meta.chunk_count = stats.get("chunks", 0)
    meta.embedder = stats.get("embedder", "hashing")
    db.commit()


@router.post("/index")
def build_index(
    ws: Workspace = Depends(get_workspace),
    services: Services = Depends(get_services),
    db: Session = Depends(get_db),
) -> dict:
    idx = services.indexer_for(ws.root_dir)
    stats = idx.build()
    _persist_meta(db, ws.id, stats)
    metrics.set_index_size(ws.id, stats["chunks"])
    return stats


@router.post("/reindex")
def reindex(
    ws: Workspace = Depends(get_workspace),
    services: Services = Depends(get_services),
    db: Session = Depends(get_db),
) -> dict:
    idx = services.indexer_for(ws.root_dir)
    stats = idx.reindex()
    _persist_meta(db, ws.id, stats)
    metrics.set_index_size(ws.id, stats["chunks"])
    return stats


@router.get("/search")
def search(
    q: str = Query(...),
    k: int = Query(6, ge=1, le=20),
    ws: Workspace = Depends(get_workspace),
    services: Services = Depends(get_services),
) -> dict:
    idx = services.indexer_for(ws.root_dir)
    results = idx.search(q, k=k)
    return {"query": q, "results": [r.to_dict() for r in results]}
