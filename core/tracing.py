"""
core/tracing.py - Lightweight request tracing and node-timing instrumentation,
                  plus optional Phoenix Cloud (OpenTelemetry) integration.

Each user turn gets a short trace_id that is attached to every log line
for that turn, making it possible to follow a single request through
intent classification, tool calls, and the final response in the logs.
This part has no external dependency and always runs.

If ENABLE_TRACING=true, maybe_init_phoenix() additionally sends real
OpenTelemetry spans to Phoenix, covering every langchain_groq call made
through core/llm_client.py. Works with either:
  - A local/self-hosted Phoenix instance (e.g. running via Docker:
    `docker run -p 6006:6006 arizephoenix/phoenix:latest`) - no API key
    needed, just set PHOENIX_COLLECTOR_ENDPOINT to its address.
  - Phoenix Cloud (https://app.phoenix.arize.com) - requires PHOENIX_API_KEY.

Requires the optional packages:
    pip install arize-phoenix-otel openinference-instrumentation-langchain
"""

from __future__ import annotations

import functools
import os
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
_PHOENIX_INIT_ATTEMPTED = False


def maybe_init_phoenix():
    """
    Best-effort Phoenix tracing init. No-op if ENABLE_TRACING is false.
    Safe to call multiple times; only initialises once per process.

    Supports two modes, auto-detected from PHOENIX_COLLECTOR_ENDPOINT:
      - Local/self-hosted (e.g. http://localhost:6006/v1/traces, the
        default - matches `docker run -p 6006:6006 arizephoenix/phoenix`):
        no API key required.
      - Phoenix Cloud (https://app.phoenix.arize.com): requires
        PHOENIX_API_KEY from Settings > API Keys on that site.

    What gets traced: `register(auto_instrument=True)` activates every
    installed OpenInference instrumentor. With `openinference-instrumentation-
    langchain` installed, every langchain_groq call (intent classifier,
    greeting, faq, database_agent, etc.) is automatically captured as a span -
    no manual span code needed in the node files.
    """
    global _PHOENIX_TRACER, _PHOENIX_INIT_ATTEMPTED

    if not app.enable_tracing:
        return None
    if _PHOENIX_INIT_ATTEMPTED:
        return _PHOENIX_TRACER
    _PHOENIX_INIT_ATTEMPTED = True

    if not app.phoenix_is_local and not app.phoenix_api_key:
        logger.warning(
            "ENABLE_TRACING is true but PHOENIX_API_KEY is not set - "
            "tracing will not be sent. Get a key from https://app.phoenix.arize.com, "
            "or point PHOENIX_COLLECTOR_ENDPOINT at a local Phoenix instance."
        )
        return None

    # register() reads PHOENIX_API_KEY from the environment itself and
    # attaches it as an auth header. Local Phoenix doesn't need or check
    # this, so only set it when we actually have one.
    if app.phoenix_api_key:
        os.environ.setdefault("PHOENIX_API_KEY", app.phoenix_api_key)

    try:
        from phoenix.otel import register

        provider = register(
            project_name=app.phoenix_project,
            endpoint=app.phoenix_endpoint,
            auto_instrument=True,
            batch=True,
        )
        _PHOENIX_TRACER = provider.get_tracer(__name__)
        logger.info(
            "Phoenix tracing enabled (mode=%s, project=%s, endpoint=%s)",
            "local" if app.phoenix_is_local else "cloud",
            app.phoenix_project,
            app.phoenix_endpoint,
        )
    except ImportError as exc:
        logger.warning(
            "ENABLE_TRACING is true but required packages are missing (%s). "
            "Install with: pip install arize-phoenix-otel openinference-instrumentation-langchain",
            exc,
        )
        _PHOENIX_TRACER = None
    except Exception as exc:
        logger.warning(
            "Phoenix tracing requested but failed to initialise (endpoint=%s): %s",
            app.phoenix_endpoint,
            exc,
        )
        _PHOENIX_TRACER = None

    return _PHOENIX_TRACER
