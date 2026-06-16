"""
graph/nodes/misc_nodes.py - Clarification, human escalation, and
general conversation nodes. Each is small enough to share one file.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from core.llm_client import get_llm, safe_invoke
from core.logging_setup import get_logger
from core.tracing import timed_node
from graph.state import GraphState
from prompts.templates import (
    CLARIFICATION_PROMPT,
    GENERAL_CONVERSATION_PROMPT,
    HUMAN_ESCALATION_PROMPT,
)

logger = get_logger(__name__)


@timed_node("clarification")
def clarification_node(state: GraphState) -> GraphState:
    prompt = CLARIFICATION_PROMPT.format(
        user_message=state.get("user_message", ""),
        conversation_history=state.get("conversation_history", "") or "(no prior messages)",
    )
    llm = get_llm()
    result = safe_invoke(
        llm,
        [HumanMessage(content=prompt)],
        fallback_text="Could you tell me a bit more about what you need help with?",
    )
    return {"final_answer": result.text, "needs_clarification": True}


@timed_node("human_escalation")
def human_escalation_node(state: GraphState) -> GraphState:
    prompt = HUMAN_ESCALATION_PROMPT.format(user_message=state.get("user_message", ""))
    llm = get_llm()
    result = safe_invoke(
        llm,
        [HumanMessage(content=prompt)],
        fallback_text=(
            "I understand you'd like to speak with a human representative. "
            "Your request has been noted and someone will follow up shortly."
        ),
    )
    return {"final_answer": result.text}


@timed_node("general_conversation")
def general_conversation_node(state: GraphState) -> GraphState:
    prompt = GENERAL_CONVERSATION_PROMPT.format(
        user_message=state.get("user_message", ""),
        conversation_history=state.get("conversation_history", "") or "(no prior messages)",
    )
    llm = get_llm()
    result = safe_invoke(
        llm,
        [HumanMessage(content=prompt)],
        fallback_text="Happy to help - what would you like to know about your insurance?",
    )
    return {"final_answer": result.text}
