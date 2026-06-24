"""ASGI middleware: request IDs, security headers, metrics, body-size cap."""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ..observability import metrics
from ..observability.logging import get_logger, set_request_id

log = get_logger("aiforge.http")

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "X-XSS-Protection": "0",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
    ),
}


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, log + time the request, and add metrics/headers."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        set_request_id(request_id)
        request.state.request_id = request_id
        start = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:  # pragma: no cover - handled by exception handlers too
            log.error("unhandled_exception", method=request.method, path=request.url.path)
            raise
        elapsed = time.monotonic() - start
        # Read the matched route to use a low-cardinality metric label.
        route = request.scope.get("route")
        path_label = getattr(route, "path", request.url.path)
        try:
            metrics.http_requests_total.labels(
                method=request.method, path=path_label, status=str(response.status_code)
            ).inc()
            metrics.http_request_latency.labels(method=request.method, path=path_label).observe(
                elapsed
            )
        except Exception:  # pragma: no cover
            pass
        response.headers["X-Request-ID"] = request_id
        for key, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(key, value)
        log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(elapsed * 1000, 2),
        )
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Content-Length exceeds the configured cap."""

    def __init__(self, app, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next) -> Response:
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > self.max_bytes:
                    return JSONResponse({"detail": "request body too large"}, status_code=413)
            except ValueError:
                pass
        return await call_next(request)
