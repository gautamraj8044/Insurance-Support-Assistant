"""
graph/builder.py - Builds and compiles the LangGraph workflow.

Flow:
    START
      -> classify_intent
           -> greeting?            -> greeting_node           -> END
           -> human_escalation?    -> human_escalation_node    -> END
           -> clarification_needed?-> clarification_node       -> END
           -> faq?                 -> faq_node                 -> END
           -> requires_tool?       -> database_agent_node      -> END
           -> else                 -> general_conversation_node-> END

This mirrors the explicit decision tree from the spec: intent detection
happens once, then a single conditional edge routes to exactly one
terminal node. There is no supervisor re-entry loop - each user turn is
one classification + one response, which is both cheaper (fewer LLM
calls) and removes the earlier "infinite re-routing" failure mode.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from core.logging_setup import get_logger
from graph.nodes.database_agent import database_agent_node
from graph.nodes.faq import faq_node
from graph.nodes.greeting import greeting_node
from graph.nodes.intent_classifier import classify_intent_node
from graph.nodes.misc_nodes import (
    clarification_node,
    general_conversation_node,
    human_escalation_node,
)
from graph.state import GraphState

logger = get_logger(__name__)


def _route_after_intent(state: GraphState) -> str:
    intent = state.get("intent", "general_conversation")

    if intent == "greeting":
        return "greeting"
    if intent == "human_escalation":
        return "human_escalation"
    if intent == "clarification_needed":
        return "clarification"
    if intent == "faq":
        return "faq"
    if state.get("requires_tool"):
        return "database_agent"
    return "general_conversation"


def build_app():
    """Construct and compile the LangGraph StateGraph."""
    workflow = StateGraph(GraphState)

    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("faq", faq_node)
    workflow.add_node("database_agent", database_agent_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("human_escalation", human_escalation_node)
    workflow.add_node("general_conversation", general_conversation_node)

    workflow.set_entry_point("classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        _route_after_intent,
        {
            "greeting": "greeting",
            "faq": "faq",
            "database_agent": "database_agent",
            "clarification": "clarification",
            "human_escalation": "human_escalation",
            "general_conversation": "general_conversation",
        },
    )

    for node in (
        "greeting",
        "faq",
        "database_agent",
        "clarification",
        "human_escalation",
        "general_conversation",
    ):
        workflow.add_edge(node, END)

    logger.info("Graph compiled successfully.")
    return workflow.compile()
