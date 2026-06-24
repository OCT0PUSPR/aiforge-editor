"""AI endpoints: inline completion, chat, and agentic (multi-file) edit.

Streaming responses use SSE with periodic heartbeats and client-disconnect
cancellation so a closed browser tab stops the (possibly expensive) generation.
Chat persists messages with token/cost accounting; edits persist reversible
history for undo.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator, Iterator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from ...ai import completion as completion_feature
from ...ai.chat import chat as chat_feature
from ...ai.diff import DiffConflict
from ...ai.edit import (
    apply_edit,
    apply_full_content,
    apply_multifile_edit,
    propose_edit,
    propose_multifile_edit,
)
from ...db import ChatSession, EditHistory, Workspace, get_db
from ...db import Message as DbMessage
from ...llm.base import estimate_cost, estimate_tokens
from ...observability import get_logger, metrics, span
from ...workspace import PathTraversalError
from ...workspace import Workspace as FsWorkspace
from ..deps import ai_rate_limit, get_fs, get_services, get_workspace
from ..schemas import (
    ChatRequest,
    CompleteRequest,
    EditApplyRequest,
    EditRequest,
    MultiEditRequest,
)
from ..services import Services

router = APIRouter(
    prefix="/api/workspaces/{workspace_id}/ai",
    tags=["ai"],
    dependencies=[Depends(ai_rate_limit)],
)

log = get_logger("aiforge.ai")

_HEARTBEAT_SECONDS = 15.0


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _sse_from_sync(
    request: Request,
    event_name: str,
    chunks: Iterator[str],
    *,
    on_text=None,
    final: Optional[dict] = None,
) -> AsyncIterator[str]:
    """Bridge a synchronous chunk iterator to an async SSE stream.

    Runs the blocking iterator in a thread, emits heartbeats during idle gaps,
    and stops early if the client disconnects.
    """
    loop = asyncio.get_event_loop()
    iterator = iter(chunks)
    sentinel = object()

    def _next():
        try:
            return next(iterator)
        except StopIteration:
            return sentinel

    last_emit = time.monotonic()
    try:
        while True:
            if await request.is_disconnected():
                log.info("sse_client_disconnected")
                break
            try:
                chunk = await asyncio.wait_for(
                    loop.run_in_executor(None, _next), timeout=_HEARTBEAT_SECONDS
                )
            except asyncio.TimeoutError:
                yield _sse("heartbeat", {"t": time.time()})
                last_emit = time.monotonic()
                continue
            if chunk is sentinel:
                break
            if chunk:
                if on_text is not None:
                    on_text(chunk)
                yield _sse(event_name, {"text": chunk})
                last_emit = time.monotonic()
        if final is not None:
            yield _sse("meta", final)
        yield _sse("done", {})
    except Exception as exc:  # pragma: no cover - defensive
        log.error("sse_error", error=str(exc))
        yield _sse("error", {"message": str(exc)})
    finally:
        _ = last_emit  # silence unused in some branches


# -- completion -------------------------------------------------------------
@router.post("/complete")
async def complete(
    request: Request,
    body: CompleteRequest,
    services: Services = Depends(get_services),
):
    backend = services.backend_for("complete")
    provider = getattr(backend, "name", "unknown")
    metrics.record_completion(provider)
    chunks = completion_feature.complete(
        backend,
        prefix=body.prefix,
        suffix=body.suffix,
        language=body.language,
        path=body.path,
        max_tokens=body.max_tokens,
        model=services.settings.model_for("complete"),
    )
    gen = _sse_from_sync(request, "token", chunks)
    return StreamingResponse(gen, media_type="text/event-stream")


# -- chat -------------------------------------------------------------------
@router.post("/chat")
async def chat(
    request: Request,
    body: ChatRequest,
    ws: Workspace = Depends(get_workspace),
    services: Services = Depends(get_services),
    db: Session = Depends(get_db),
):
    backend = services.backend_for("chat")
    provider = getattr(backend, "name", "unknown")
    model = services.settings.model_for("chat")
    indexer = services.indexer_for(ws.root_dir)

    stream, results = chat_feature(
        backend,
        indexer,
        question=body.question,
        open_path=body.open_path,
        open_content=body.open_content,
        history=[(t.role, t.content) for t in body.history],
        top_k=body.top_k,
        model=model,
    )

    collected: List[str] = []

    def _on_text(text: str) -> None:
        collected.append(text)

    refs = [r.to_dict() for r in results]

    # Persist messages + accounting after the stream completes. We do this in a
    # callback the generator triggers when done.
    workspace_id = ws.id

    async def gen():
        async for piece in _sse_from_sync(
            request, "token", stream, on_text=_on_text, final={"references": refs}
        ):
            yield piece
        # Stream finished (or client disconnected): record what we produced.
        answer = "".join(collected)
        in_tok = estimate_tokens(body.question + "".join(c.get("text", "") for c in refs))
        out_tok = estimate_tokens(answer)
        cost = estimate_cost(model, in_tok, out_tok)
        metrics.record_chat(provider, in_tok, out_tok, cost)
        try:
            _persist_chat(db, workspace_id, body, answer, in_tok, out_tok, cost, provider)
        except Exception as exc:  # pragma: no cover - persistence best effort
            log.warning("chat_persist_failed", error=str(exc))

    return StreamingResponse(gen(), media_type="text/event-stream")


def _persist_chat(
    db: Session,
    workspace_id: str,
    body: ChatRequest,
    answer: str,
    in_tok: int,
    out_tok: int,
    cost: float,
    provider: str,
) -> None:
    session_id = body.session_id
    if session_id:
        session = db.get(ChatSession, session_id)
        if session is None or session.workspace_id != workspace_id:
            session = None
    else:
        session = None
    if session is None:
        session = ChatSession(workspace_id=workspace_id, title=body.question[:60])
        db.add(session)
        db.flush()
    db.add(DbMessage(session_id=session.id, role="user", content=body.question))
    db.add(
        DbMessage(
            session_id=session.id,
            role="assistant",
            content=answer,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            provider=provider,
        )
    )
    db.commit()


# -- edit: propose ----------------------------------------------------------
@router.post("/edit")
def edit(
    body: EditRequest,
    fs: FsWorkspace = Depends(get_fs),
    services: Services = Depends(get_services),
):
    backend = services.backend_for("edit")
    try:
        with span("ai.edit.propose", path=body.path):
            proposal = propose_edit(
                backend,
                fs,
                path=body.path,
                instruction=body.instruction,
                model=services.settings.model_for("edit"),
            )
    except PathTraversalError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    metrics.record_edit(applied=False)
    out = proposal.to_dict()
    out["instruction"] = body.instruction
    return JSONResponse(out)


@router.post("/edit/multi")
def edit_multi(
    body: MultiEditRequest,
    fs: FsWorkspace = Depends(get_fs),
    services: Services = Depends(get_services),
):
    backend = services.backend_for("edit")
    try:
        proposal = propose_multifile_edit(
            backend,
            fs,
            paths=body.paths,
            instruction=body.instruction,
            model=services.settings.model_for("edit"),
        )
    except PathTraversalError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    metrics.record_edit(applied=False)
    out = proposal.to_dict()
    out["instruction"] = body.instruction
    return JSONResponse(out)


# -- edit: apply ------------------------------------------------------------
@router.post("/edit/apply")
def edit_apply(
    body: EditApplyRequest,
    ws: Workspace = Depends(get_workspace),
    fs: FsWorkspace = Depends(get_fs),
    db: Session = Depends(get_db),
):
    try:
        if body.multifile and body.diff is not None:
            result = apply_multifile_edit(fs, diff=body.diff)
            for path in result.applied:
                db.add(
                    EditHistory(
                        workspace_id=ws.id,
                        path=path,
                        instruction=body.instruction,
                        forward_diff=body.diff,
                        reverse_diff=result.reverse_diff,
                    )
                )
            db.commit()
            metrics.record_edit(applied=True)
            return JSONResponse(result.to_dict())

        if body.path is None:
            raise HTTPException(status_code=400, detail="path required")

        if body.new_content is not None:
            res = apply_full_content(fs, path=body.path, new_content=body.new_content)
        elif body.diff is not None:
            res = apply_edit(
                fs,
                path=body.path,
                diff=body.diff,
                expected_original=body.expected_original,
            )
        else:
            raise HTTPException(status_code=400, detail="provide 'diff' or 'new_content'")

        # Record forward/reverse for undo.
        forward = body.diff or ""
        db.add(
            EditHistory(
                workspace_id=ws.id,
                path=body.path,
                instruction=body.instruction,
                forward_diff=forward,
                reverse_diff=res.reverse_diff,
            )
        )
        db.commit()
        metrics.record_edit(applied=True)
        return JSONResponse(res.to_dict())
    except PathTraversalError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except DiffConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))


# -- edit: history + undo ---------------------------------------------------
@router.get("/edit/history")
def edit_history(
    ws: Workspace = Depends(get_workspace),
    db: Session = Depends(get_db),
):
    from sqlalchemy import select

    rows = (
        db.execute(
            select(EditHistory)
            .where(EditHistory.workspace_id == ws.id)
            .order_by(EditHistory.created_at.desc())
            .limit(50)
        )
        .scalars()
        .all()
    )
    return {
        "history": [
            {
                "id": r.id,
                "path": r.path,
                "instruction": r.instruction,
                "applied": r.applied,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]
    }


@router.post("/edit/undo/{edit_id}")
def edit_undo(
    edit_id: str,
    ws: Workspace = Depends(get_workspace),
    fs: FsWorkspace = Depends(get_fs),
    db: Session = Depends(get_db),
):
    record = db.get(EditHistory, edit_id)
    if record is None or record.workspace_id != ws.id:
        raise HTTPException(status_code=404, detail="edit not found")
    if not record.applied:
        raise HTTPException(status_code=409, detail="edit already undone")
    try:
        apply_multifile_edit(fs, diff=record.reverse_diff)
    except DiffConflict as exc:
        raise HTTPException(status_code=409, detail=f"cannot undo: {exc}")
    record.applied = False
    db.commit()
    return {"id": edit_id, "undone": True}
