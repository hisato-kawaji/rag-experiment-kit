"""Optional OpenTelemetry → Phoenix tracing.

Off by default. Enable with `PHOENIX_ENABLED=true` and `task trace-up`. The
helper `traced_span(name, **attrs)` is a no-op context manager unless tracing
has been initialized, so call sites stay safe whether the user opted in or not.

Spans worth attaching:
- pipeline answer entry (`baseline.answer`, `graphrag.global`, ...)
- vector store search
- LLM completions
- inner loops with high fan-out (e.g. graphrag global map calls)
"""

from __future__ import annotations

from contextlib import contextmanager, suppress
from typing import Any

from .config import Settings
from .logging import log

_tracer: Any | None = None


def setup_tracing(settings: Settings | None = None) -> None:
    """Initialize OTLP exporter pointed at Phoenix. Idempotent."""
    global _tracer
    if _tracer is not None:
        return
    s = settings or Settings()
    if not s.phoenix_enabled:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        log.warning(
            "phoenix.import-failed",
            hint="Install with: uv sync --extra phoenix",
        )
        return

    endpoint = s.phoenix_endpoint.rstrip("/") + "/v1/traces"
    provider = TracerProvider(resource=Resource.create({"service.name": "rag-experiment-kit"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("rag")
    log.info("phoenix.tracing.enabled", endpoint=endpoint)


@contextmanager
def traced_span(name: str, **attrs: Any):
    """No-op when tracing is disabled, real OTel span otherwise."""
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            with suppress(Exception):
                span.set_attribute(k, v)
        yield span


__all__ = ["setup_tracing", "traced_span"]
