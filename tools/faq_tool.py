"""
tools/faq_tool.py - ChromaDB-backed FAQ retrieval.

Wraps the vector store lookup behind a single function so graph nodes
don't need to know about ChromaDB internals. Returns a formatted
context string ready to be dropped into a prompt, plus the raw matches
for logging/debugging.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List

import chromadb

from core.exceptions import VectorStoreError
from core.logging_setup import get_logger
from core.settings import faq, paths

logger = get_logger(__name__)


@dataclass
class FAQMatch:
    question: str
    answer: str
    score: float


@lru_cache(maxsize=1)
def _get_collection():
    try:
        client = chromadb.PersistentClient(path=paths.chroma_path)
        return client.get_or_create_collection(name=paths.chroma_collection)
    except Exception as exc:
        raise VectorStoreError(f"Could not initialise ChromaDB: {exc}") from exc


def is_faq_store_ready() -> bool:
    """Check whether the FAQ collection has any documents loaded."""
    try:
        collection = _get_collection()
        return collection.count() > 0
    except VectorStoreError:
        return False


def search_faqs(query: str, top_k: int = None) -> List[FAQMatch]:
    """
    Search the FAQ vector store for the top_k most relevant entries.
    Returns an empty list (never raises) if the store is unavailable
    or has no matches - callers should handle the empty case gracefully.
    """
    top_k = top_k or faq.top_k
    try:
        collection = _get_collection()
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
    except VectorStoreError as exc:
        logger.warning("FAQ search unavailable: %s", exc)
        return []
    except Exception:
        logger.exception("Unexpected error during FAQ search")
        return []

    matches: List[FAQMatch] = []
    metadatas = results.get("metadatas") or []
    distances = results.get("distances") or []
    if metadatas and metadatas[0]:
        for meta, dist in zip(metadatas[0], distances[0]):
            matches.append(
                FAQMatch(
                    question=meta.get("question", ""),
                    answer=meta.get("answer", ""),
                    score=float(dist),
                )
            )
    return matches


def format_faq_context(matches: List[FAQMatch]) -> str:
    """Render FAQ matches into a prompt-ready context block."""
    if not matches:
        return "No relevant FAQs were found in the knowledge base."

    blocks = []
    for i, m in enumerate(matches, start=1):
        blocks.append(f"FAQ {i} (relevance score: {m.score:.3f})\nQ: {m.question}\nA: {m.answer}")
    return "\n\n".join(blocks)
