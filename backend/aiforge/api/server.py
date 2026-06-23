"""FastAPI application exposing the aiforge AI editor backend.

Endpoints
---------
- ``GET  /health``           liveness + backend/index status
- ``GET  /api/tree``         workspace file tree
- ``GET  /api/file``         read a file (``?path=``)
- ``PUT  /api/file``         save a file
- ``POST /api/file``         create a file
- ``DELETE /api/file``       delete a file (``?path=``)
- ``POST /api/complete``     inline completion (SSE stream)
- ``POST /api/chat``         codebase chat (SSE stream, with references)
- ``POST /api/edit``         propose an agentic edit (returns a diff)
- ``POST /api/edit/apply``   apply a proposed diff to the workspace
- ``POST /api/index``        (re)build the RAG index
- ``GET  /api/search``       RAG search (``?q=&k=``)

The SSE contract is a stream of ``event:``/``data:`` lines; data is JSON. The
frontend ``client.ts`` parses the same shape. The dev frontend origin is
allowed via CORS; in production the built frontend is served from ``/``.
"""
from __future__ import annotations

import json
from typing import Iterator, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - server requires fastapi
    raise RuntimeError(
        "FastAPI is required to run the server. Install backend requirements."
    ) from exc

from ..ai import chat as chat_feature
from ..ai import completion as completion_feature
from ..ai.diff import DiffError
from ..ai.edit import apply_edit, apply_full_content, propose_edit
from ..config import Settings, get_settings
from ..llm import get_backend
from ..rag.indexer import RagIndexer
from ..workspace.files import (
    NotFoundError,
    PathTraversalError,
    Workspace,
    WorkspaceError,
)


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------
class FileSave(BaseModel):
    path: str
    content: str


class FileCreate(BaseModel):
    path: str
    content: str = ""


class CompleteRequest(BaseModel):
    prefix: str
    suffix: str = ""
    language: str = ""
    path: str = ""
    max_tokens: int = 256


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    open_path: str = ""
    open_content: str = ""
    history: List[ChatTurn] = []
    top_k: int = 6


class EditRequest(BaseModel):
    path: str
    instruction: str


class EditApplyRequest(BaseModel):
    path: str
    diff: Optional[str] = None
    new_content: Optional[str] = None
    expected_original: Optional[str] = None


# --------------------------------------------------------------------------
# SSE helpers (shared contract with the frontend client)
# --------------------------------------------------------------------------
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_tokens(event_name: str, chunks: Iterator[str], *, final: Optional[dict] = None):
    def gen() -> Iterator[str]:
        try:
            for chunk in chunks:
                if chunk:
                    yield _sse(event_name, {"text": chunk})
            if final is not None:
                yield _sse("meta", final)
            yield _sse("done", {})
        except Exception as exc:  # surface backend errors to the client
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(gen(), media_type="text/event-stream")


# --------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------
def create_app(settings: Optional[Settings] = None) -> "FastAPI":
    settings = settings or get_settings()
    workspace = Workspace(settings.resolved_workspace_root())
    indexer = RagIndexer(
        workspace,
        embed_dim=settings.rag_embed_dim,
        chunk_lines=settings.rag_chunk_lines,
        chunk_overlap=settings.rag_chunk_overlap,
        use_sentence_transformers=settings.rag_use_sentence_transformers,
    )

    app = FastAPI(title="aiforge-editor", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list() or ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Stash shared state for handlers and tests.
    app.state.settings = settings
    app.state.workspace = workspace
    app.state.indexer = indexer

    def backend_for(feature: str):
        return get_backend(settings.backend, model=settings.model_for(feature))

    # -- health ----------------------------------------------------------
    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "backend": settings.backend,
            "workspace": str(workspace.root),
            "indexed_files": indexer.file_count,
            "indexed_chunks": indexer.chunk_count,
        }

    # -- workspace -------------------------------------------------------
    @app.get("/api/tree")
    def tree(path: str = Query("")) -> dict:
        try:
            return workspace.tree(path).to_dict()
        except NotFoundError:
            raise HTTPException(status_code=404, detail="path not found")
        except PathTraversalError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/file")
    def read_file(path: str = Query(...)) -> dict:
        try:
            return {"path": path, "content": workspace.read(path)}
        except NotFoundError:
            raise HTTPException(status_code=404, detail="file not found")
        except PathTraversalError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.put("/api/file")
    def save_file(body: FileSave) -> dict:
        try:
            workspace.write(body.path, body.content)
            return {"path": body.path, "saved": True}
        except PathTraversalError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/file")
    def create_file(body: FileCreate) -> dict:
        try:
            workspace.create(body.path, body.content)
            return {"path": body.path, "created": True}
        except PathTraversalError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except WorkspaceError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.delete("/api/file")
    def delete_file(path: str = Query(...)) -> dict:
        try:
            workspace.delete(path)
            return {"path": path, "deleted": True}
        except NotFoundError:
            raise HTTPException(status_code=404, detail="file not found")
        except PathTraversalError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except WorkspaceError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # -- RAG -------------------------------------------------------------
    @app.post("/api/index")
    def build_index() -> dict:
        return indexer.build()

    @app.get("/api/search")
    def search(q: str = Query(...), k: int = Query(6)) -> dict:
        results = indexer.search(q, k=k)
        return {"query": q, "results": [r.to_dict() for r in results]}

    # -- AI: completion (SSE) -------------------------------------------
    @app.post("/api/complete")
    def complete(body: CompleteRequest):
        backend = backend_for("complete")
        chunks = completion_feature.complete(
            backend,
            prefix=body.prefix,
            suffix=body.suffix,
            language=body.language,
            path=body.path,
            max_tokens=body.max_tokens,
            model=settings.model_for("complete"),
        )
        return _stream_tokens("token", chunks)

    # -- AI: chat (SSE) --------------------------------------------------
    @app.post("/api/chat")
    def chat(body: ChatRequest):
        backend = backend_for("chat")
        stream, results = chat_feature.chat(
            backend,
            indexer,
            question=body.question,
            open_path=body.open_path,
            open_content=body.open_content,
            history=[(t.role, t.content) for t in body.history],
            top_k=body.top_k,
            model=settings.model_for("chat"),
        )
        return _stream_tokens(
            "token",
            stream,
            final={"references": [r.to_dict() for r in results]},
        )

    # -- AI: edit propose ------------------------------------------------
    @app.post("/api/edit")
    def edit(body: EditRequest):
        backend = backend_for("edit")
        try:
            proposal = propose_edit(
                backend,
                workspace,
                path=body.path,
                instruction=body.instruction,
                model=settings.model_for("edit"),
            )
        except PathTraversalError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return JSONResponse(proposal.to_dict())

    # -- AI: edit apply --------------------------------------------------
    @app.post("/api/edit/apply")
    def edit_apply(body: EditApplyRequest):
        try:
            if body.new_content is not None:
                result = apply_full_content(
                    workspace, path=body.path, new_content=body.new_content
                )
            elif body.diff is not None:
                result = apply_edit(
                    workspace,
                    path=body.path,
                    diff=body.diff,
                    expected_original=body.expected_original,
                )
            else:
                raise HTTPException(
                    status_code=400, detail="provide either 'diff' or 'new_content'"
                )
        except PathTraversalError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except DiffError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return JSONResponse(result.to_dict())

    # -- production: serve the built frontend ---------------------------
    _maybe_mount_frontend(app, settings)

    return app


def _maybe_mount_frontend(app: "FastAPI", settings: Settings) -> None:
    """Serve the built SPA from ``/`` when AIFORGE_FRONTEND_DIST is set."""
    dist = settings.frontend_dist
    if not dist:
        return
    import os

    if not os.path.isdir(dist):
        return
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")


# Module-level app for ``uvicorn aiforge.api.server:app``.
app = None
try:  # Construct eagerly so `uvicorn ...:app` works; tolerate import-time issues.
    app = create_app()
except Exception:  # pragma: no cover - defensive; create_app called in tests too
    app = None


def run_cli() -> None:  # pragma: no cover - thin CLI shim
    """Console-script entrypoint: ``aiforge-server``."""
    import os

    import uvicorn

    host = os.environ.get("AIFORGE_HOST", "0.0.0.0")
    port = int(os.environ.get("AIFORGE_PORT", "8000"))
    uvicorn.run("aiforge.api.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":  # pragma: no cover
    run_cli()
