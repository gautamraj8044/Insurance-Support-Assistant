"""
graph/nodes/greeting.py - Direct greeting response, no tools.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from core.llm_client import get_llm, safe_invoke
from core.logging_setup import get_logger
from core.tracing import timed_node
from graph.state import GraphState
from prompts.templates import GREETING_RESPONSE_PROMPT

logger = get_logger(__name__)

_FALLBACK_GREETING = (
    "Hello! Welcome to Insurance Support. I'm here to help with policies, "
    "claims, premiums, renewals, and coverage questions. How can I assist you today?"
)


@timed_node("greeting")
def greeting_node(state: GraphState) -> GraphState:
    prompt = GREETING_RESPONSE_PROMPT.format(user_message=state.get("user_message", ""))
    llm = get_llm()
    result = safe_invoke(llm, [HumanMessage(content=prompt)], fallback_text=_FALLBACK_GREETING)
    return {"final_answer": result.text}
