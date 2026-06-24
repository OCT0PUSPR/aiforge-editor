"""Optional OpenTelemetry tracing.

OTel is fully optional: if the packages aren't installed (the default), the
``span`` context manager is a no-op, so the rest of the app uses it
unconditionally without taking a hard dependency.
"""

from __future__ import annotations

import contextlib
from typing import Iterator, Optional

try:  # pragma: no cover - exercised only when OTel is installed
    from opentelemetry import trace

    _tracer = trace.get_tracer("aiforge")
    _HAVE_OTEL = True
except Exception:  # noqa: BLE001
    _tracer = None
    _HAVE_OTEL = False


@contextlib.contextmanager
def span(name: str, **attributes) -> Iterator[Optional[object]]:
    """Start an OTel span if available; otherwise a no-op context manager."""
    if not _HAVE_OTEL or _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as current:  # pragma: no cover
        for key, value in attributes.items():
            try:
                current.set_attribute(key, value)
            except Exception:  # noqa: BLE001
                pass
        yield current
