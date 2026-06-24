"""Observability: structured logging, Prometheus metrics, optional OTel spans."""

from . import metrics
from .logging import configure_logging, get_logger, set_request_id
from .tracing import span

__all__ = ["configure_logging", "get_logger", "metrics", "set_request_id", "span"]
