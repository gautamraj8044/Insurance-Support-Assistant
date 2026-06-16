"""
core/llm_client.py - ChatGroq client wrapper.

Centralises Groq LLM access so the rest of the app never imports
langchain_groq directly. Provides:
  - get_llm() / get_intent_llm(): cached ChatGroq instances
  - safe_invoke(): never raises - returns a normalised result object
  - extract_text(): normalises .content (str or list-of-blocks) to str
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Optional

from langchain_groq import ChatGroq

from core.exceptions import LLMUnavailableError
from core.logging_setup import get_logger
from core.settings import groq

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def _build_client(model: str) -> ChatGroq:
    if not groq.api_key:
        raise LLMUnavailableError(
            "GROQ_API_KEY is not configured. Set it in your .env file."
        )
    return ChatGroq(
        api_key=groq.api_key,
        model=model,
        temperature=groq.temperature,
        max_retries=groq.max_retries,
        timeout=groq.request_timeout,
    )


def get_llm(model: Optional[str] = None) -> ChatGroq:
    """Return a cached ChatGroq instance for the main agent model."""
    return _build_client(model or groq.model)


def get_intent_llm() -> ChatGroq:
    """Return a cached ChatGroq instance for the lightweight intent classifier."""
    return _build_client(groq.intent_model)


def extract_text(content: Any) -> str:
    """
    Normalise an LLM message's .content field to plain text.
    Groq/LangChain usually returns a string, but some providers/models
    return a list of content blocks (e.g. [{"type": "text", "text": "..."}]).
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif item.get("type") == "text" and "content" in item:
                    parts.append(str(item["content"]))
        return "".join(parts)
    return str(content)


@dataclass
class LLMResult:
    text: str
    tool_calls: list
    succeeded: bool
    duration_ms: float
    error: Optional[str] = None


_FALLBACK_MESSAGE = (
    "I'm having trouble reaching the AI service right now. "
    "Please try again in a moment."
)
_RATE_LIMIT_MESSAGE = (
    "Our AI service has reached its usage limit right now. "
    "Please wait a moment and try again."
)


def safe_invoke(
    llm: ChatGroq,
    messages: list,
    fallback_text: Optional[str] = None,
) -> LLMResult:
    """
    Invoke an LLM (optionally bound with tools) without ever raising.
    Returns an LLMResult with succeeded=False and a friendly message
    on any failure (rate limit, timeout, network error, etc).
    """
    start = time.time()
    try:
        response = llm.invoke(messages)
        duration_ms = (time.time() - start) * 1000
        text = extract_text(response.content)
        tool_calls = getattr(response, "tool_calls", []) or []
        logger.info("LLM call succeeded in %.0fms", duration_ms)
        return LLMResult(
            text=text,
            tool_calls=tool_calls,
            succeeded=True,
            duration_ms=duration_ms,
        )
    except Exception as exc:
        duration_ms = (time.time() - start) * 1000
        error_text = str(exc)
        is_rate_limit = "429" in error_text or "rate_limit" in error_text.lower()

        if is_rate_limit:
            logger.error("Groq rate limit exceeded after %.0fms: %s", duration_ms, error_text)
            message = fallback_text or _RATE_LIMIT_MESSAGE
        else:
            logger.exception("LLM call failed after %.0fms", duration_ms)
            message = fallback_text or _FALLBACK_MESSAGE

        return LLMResult(
            text=message,
            tool_calls=[],
            succeeded=False,
            duration_ms=duration_ms,
            error=error_text,
        )
