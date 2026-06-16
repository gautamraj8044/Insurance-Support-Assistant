"""
graph/nodes/faq.py - FAQ / general knowledge response node (RAG over ChromaDB).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from core.llm_client import get_llm, safe_invoke
from core.logging_setup import get_logger
from core.tracing import timed_node
from graph.state import GraphState
from prompts.templates import FAQ_AGENT_PROMPT
from tools.faq_tool import format_faq_context, search_faqs

logger = get_logger(__name__)

_FALLBACK_TEXT = (
    "I don't have specific information on that right now, but I'd be happy "
    "to connect you with a representative who can help further."
)


@timed_node("faq")
def faq_node(state: GraphState) -> GraphState:
    user_message = state.get("user_message", "")
    conversation_history = state.get("conversation_history", "")

    matches = search_faqs(user_message)
    faq_context = format_faq_context(matches)
    logger.info("FAQ search returned %d match(es)", len(matches))

    prompt = FAQ_AGENT_PROMPT.format(
        user_message=user_message,
        conversation_history=conversation_history or "(no prior messages)",
        faq_context=faq_context,
    )

    llm = get_llm()
    result = safe_invoke(llm, [HumanMessage(content=prompt)], fallback_text=_FALLBACK_TEXT)
    return {"final_answer": result.text}
