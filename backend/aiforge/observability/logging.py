"""Structured logging via structlog (JSON in prod, pretty in dev)."""

from __future__ import annotations

import contextvars
import logging
import sys
from typing import Optional

import structlog

# Request-id is propagated through a context var so every log line in a request
# carries it without threading it manually.
request_id_var: "contextvars.ContextVar[str]" = contextvars.ContextVar("request_id", default="-")


def _add_request_id(_logger, _name, event_dict):
    event_dict.setdefault("request_id", request_id_var.get())
    return event_dict


def configure_logging(*, json_logs: bool = True, level: str = "info") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=log_level)

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_request_id,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None):
    return structlog.get_logger(name)


def set_request_id(request_id: str) -> None:
    request_id_var.set(request_id)
