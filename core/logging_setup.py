"""
core/logging_setup.py - Structured logging configuration.

Provides a single configure_logging() entry point and a get_logger()
helper so every module gets a consistently formatted, ASCII-only logger
(no emojis - they break on some Windows terminals/encodings).
"""

from __future__ import annotations

import logging
import os
import sys

from core.settings import paths

_CONFIGURED = False


class _RequestContextFilter(logging.Filter):
    """Attaches a request/trace id to every log record, if present."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "trace_id"):
            record.trace_id = "-"
        return True


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotently configure root logging handlers (console + file)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    os.makedirs(paths.log_dir, exist_ok=True)
    log_file = os.path.join(paths.log_dir, "app.log")

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | [trace=%(trace_id)s] | %(message)s"
    formatter = logging.Formatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_RequestContextFilter())

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_RequestContextFilter())

    # Avoid duplicate handlers if re-imported (e.g. Streamlit hot reload)
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
