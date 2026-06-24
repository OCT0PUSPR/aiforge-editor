"""FastAPI application factory for the aiforge editor backend.

Wires together: structured logging, the services container, DB init, CORS, the
security/observability middleware, all routers (auth, workspaces, files, rag,
ai), global error handlers, ``/health``, ``/ready``, ``/metrics``, and
(optionally) serving the built frontend in production.

The app is fully multi-tenant: every workspace is sandboxed to its own
filesystem root and owned by an authenticated user.
"""

from __future__ import annotations

import contextlib
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from sqlalchemy import text

from ..config import Settings, get_settings
from ..db import get_engine, init_db
from ..observability import configure_logging, get_logger, metrics
from ..workspace import WorkspaceError
from .middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from .routers import ai, auth, files, rag, workspaces
from .services import Services

log = get_logger("aiforge.server")


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    log.info("startup", backend=settings.backend, db=settings.database_url.split("://")[0])
    init_db()
    app.state.ready = True
    try:
        yield
    finally:
        # Graceful shutdown: dispose DB connections.
        app.state.ready = False
        with contextlib.suppress(Exception):
            get_engine().dispose()
        log.info("shutdown")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(json_logs=settings.log_json, level=settings.log_level)

    app = FastAPI(
        title="aiforge-editor",
        version="0.2.0",
        description="AI-native code editor backend: completion, RAG chat, agentic edits.",
        lifespan=_lifespan,
    )
    app.state.settings = settings
    app.state.services = Services(settings)
    app.state.ready = False

    # -- middleware (order matters: outermost first) ---------------------
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list() or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    # -- routers ---------------------------------------------------------
    app.include_router(auth.router)
    app.include_router(workspaces.router)
    app.include_router(workspaces.keys_router)
    app.include_router(files.router)
    app.include_router(rag.router)
    app.include_router(ai.router)

    # -- error handlers --------------------------------------------------
    @app.exception_handler(WorkspaceError)
    async def _workspace_error(_request: Request, exc: WorkspaceError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception):
        log.error("internal_error", error=str(exc), type=type(exc).__name__)
        return JSONResponse({"detail": "internal server error"}, status_code=500)

    # -- health / readiness / metrics ------------------------------------
    @app.get("/health", tags=["ops"])
    def health() -> dict:
        return {"status": "ok", "backend": settings.backend, "version": app.version}

    @app.get("/ready", tags=["ops"])
    def ready() -> Response:
        # Ready means: lifespan finished AND the DB answers.
        ok = bool(getattr(app.state, "ready", False))
        db_ok = True
        try:
            with get_engine().connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            db_ok = False
        status_code = 200 if (ok and db_ok) else 503
        return JSONResponse({"ready": ok and db_ok, "db": db_ok}, status_code=status_code)

    @app.get("/metrics", tags=["ops"])
    def prometheus_metrics() -> Response:
        return PlainTextResponse(
            content=metrics.render_metrics(),
            media_type=metrics.METRICS_CONTENT_TYPE,
        )

    _maybe_mount_frontend(app, settings)
    return app


def _maybe_mount_frontend(app: FastAPI, settings: Settings) -> None:
    """Serve the built SPA from ``/`` when AIFORGE_FRONTEND_DIST is set."""
    import os

    dist = settings.frontend_dist
    if not dist or not os.path.isdir(dist):
        return
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")


# Module-level app for ``uvicorn aiforge.api.server:app``.
app = None
try:  # Construct eagerly so the ASGI string target works.
    app = create_app()
except Exception:  # pragma: no cover - defensive
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
