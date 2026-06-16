"""
core/tracing.py - Lightweight request tracing and node-timing instrumentation.

Each user turn gets a short trace_id that is attached to every log line
for that turn, making it possible to follow a single request through
intent classification, tool calls, and the final response in the logs.

This intentionally has no external dependency (no Phoenix/OTel) so the
app works fully offline. If ENABLE_TRACING is set and `arize-phoenix`
is installed, a best-effort OpenTelemetry exporter is layered on top.
"""

from __future__ import annotations

import functools
import time
import uuid
from contextvars import ContextVar
from typing import Callable

from core.logging_setup import get_logger
from core.settings import app

logger = get_logger(__name__)

_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


def new_trace_id() -> str:
    return uuid.uuid4().hex[:8]


def set_trace_id(trace_id: str) -> None:
    _trace_id_var.set(trace_id)


def get_trace_id() -> str:
    return _trace_id_var.get()


class _TraceIdLogFilter:
    """Injects the current trace id into every log record."""

    def filter(self, record):
        record.trace_id = get_trace_id()
        return True


def install_trace_filter() -> None:
    import logging

    f = _TraceIdLogFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(f)


def timed_node(name: str) -> Callable:
    """
    Decorator for LangGraph node functions. Logs entry/exit and duration,
    tagged with the current trace id, without changing the function's
    return value.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            logger.info("[%s] node started", name)
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start) * 1000
                logger.info("[%s] node finished in %.0fms", name, duration_ms)
                return result
            except Exception:
                duration_ms = (time.time() - start) * 1000
                logger.exception("[%s] node failed after %.0fms", name, duration_ms)
                raise

        return wrapper

    return decorator


_PHOENIX_TRACER = None


def maybe_init_phoenix():
    """Best-effort Phoenix/OpenTelemetry init. No-op if disabled or unavailable."""
    global _PHOENIX_TRACER
    if not app.enable_tracing:
        return None
    try:
        from phoenix.otel import register

        provider = register(
            project_name=app.phoenix_project,
            endpoint=app.phoenix_endpoint,
            auto_instrument=True,
        )
        _PHOENIX_TRACER = provider.get_tracer(__name__)
        logger.info("Phoenix tracing enabled (endpoint=%s)", app.phoenix_endpoint)
    except Exception as exc:
        logger.warning("Phoenix tracing requested but unavailable: %s", exc)
        _PHOENIX_TRACER = None
    return _PHOENIX_TRACER
