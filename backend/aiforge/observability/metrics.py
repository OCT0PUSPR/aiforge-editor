"""Prometheus metrics for aiforge.

Counters/histograms for completions, chat token/cost (by provider), edits
applied, index size, and request latency. The ASGI middleware records HTTP
latency; the AI layer records token/cost. A `/metrics` endpoint exposes the
registry.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# A dedicated registry so tests can construct it without polluting the default.
REGISTRY = CollectorRegistry()

http_requests_total = Counter(
    "aiforge_http_requests_total",
    "HTTP requests",
    ["method", "path", "status"],
    registry=REGISTRY,
)
http_request_latency = Histogram(
    "aiforge_http_request_latency_seconds",
    "HTTP request latency",
    ["method", "path"],
    registry=REGISTRY,
)
completions_total = Counter(
    "aiforge_completions_total",
    "Inline completions served",
    ["provider"],
    registry=REGISTRY,
)
chat_requests_total = Counter(
    "aiforge_chat_requests_total",
    "Chat requests served",
    ["provider"],
    registry=REGISTRY,
)
chat_tokens_total = Counter(
    "aiforge_chat_tokens_total",
    "Chat tokens by provider and direction",
    ["provider", "direction"],
    registry=REGISTRY,
)
chat_cost_usd_total = Counter(
    "aiforge_chat_cost_usd_total",
    "Estimated chat cost (USD) by provider",
    ["provider"],
    registry=REGISTRY,
)
edits_applied_total = Counter(
    "aiforge_edits_applied_total",
    "Agentic edits applied",
    registry=REGISTRY,
)
edits_proposed_total = Counter(
    "aiforge_edits_proposed_total",
    "Agentic edits proposed",
    registry=REGISTRY,
)
index_size_chunks = Gauge(
    "aiforge_index_size_chunks",
    "RAG index size in chunks",
    ["workspace"],
    registry=REGISTRY,
)
ai_latency = Histogram(
    "aiforge_ai_latency_seconds",
    "AI feature latency",
    ["feature", "provider"],
    registry=REGISTRY,
)


def render_metrics() -> bytes:
    return generate_latest(REGISTRY)


METRICS_CONTENT_TYPE = CONTENT_TYPE_LATEST


def record_completion(provider: str) -> None:
    completions_total.labels(provider=provider or "unknown").inc()


def record_chat(provider: str, input_tokens: int, output_tokens: int, cost: float) -> None:
    p = provider or "unknown"
    chat_requests_total.labels(provider=p).inc()
    chat_tokens_total.labels(provider=p, direction="input").inc(input_tokens)
    chat_tokens_total.labels(provider=p, direction="output").inc(output_tokens)
    chat_cost_usd_total.labels(provider=p).inc(cost)


def record_edit(applied: bool) -> None:
    if applied:
        edits_applied_total.inc()
    else:
        edits_proposed_total.inc()


def set_index_size(workspace: str, chunks: int) -> None:
    index_size_chunks.labels(workspace=workspace).set(chunks)
