"""
graph/nodes/intent_classifier.py - Lightweight intent classification node.

Uses a small, fast LLM call (separate from the main response-generation
calls) to classify the user's message into one of the supported intents
before any tool is invoked. This is the entry point of the graph.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from core.llm_client import get_intent_llm, safe_invoke
from core.logging_setup import get_logger
from core.tracing import timed_node
from graph.entity_extraction import extract_claim_id, extract_customer_id, extract_policy_number
from graph.state import GraphState
from prompts.templates import INTENT_CLASSIFIER_PROMPT

logger = get_logger(__name__)

_VALID_INTENTS = {
    "greeting",
    "faq",
    "policy_info",
    "claim_status",
    "premium_info",
    "renewal_info",
    "customer_lookup",
    "human_escalation",
    "clarification_needed",
    "general_conversation",
}

_TOOL_REQUIRING_INTENTS = {
    "policy_info",
    "claim_status",
    "premium_info",
    "renewal_info",
    "customer_lookup",
}


def _parse_json_response(raw_text: str) -> dict:
    """Strip markdown fences (if any) and parse JSON, with a safe fallback."""
    clean = raw_text.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()
    try:
        return json.loads(clean)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Intent classifier returned non-JSON output: %r", raw_text[:200])
        return {}


@timed_node("intent_classifier")
def classify_intent_node(state: GraphState) -> GraphState:
    user_message = state.get("user_message", "")
    conversation_history = state.get("conversation_history", "")

    # Opportunistically extract entities from the latest message - this
    # lets later nodes skip asking for IDs the user already provided.
    updates: dict = {}
    if not state.get("policy_number"):
        found = extract_policy_number(user_message)
        if found:
            updates["policy_number"] = found
    if not state.get("customer_id"):
        found = extract_customer_id(user_message)
        if found:
            updates["customer_id"] = found
    if not state.get("claim_id"):
        found = extract_claim_id(user_message)
        if found:
            updates["claim_id"] = found

    prompt = INTENT_CLASSIFIER_PROMPT.format(
        conversation_history=conversation_history or "(no prior messages)",
        user_message=user_message,
    )

    llm = get_intent_llm()
    result = safe_invoke(llm, [HumanMessage(content=prompt)])

    if not result.succeeded:
        # If the classifier itself fails, degrade to a safe default rather
        # than blocking the whole turn - route to FAQ/general handling.
        logger.warning("Intent classification failed - defaulting to faq intent.")
        updates.update(
            {
                "intent": "faq",
                "intent_confidence": 0.0,
                "requires_tool": False,
                "error": result.text,
            }
        )
        return updates

    parsed = _parse_json_response(result.text)
    intent = parsed.get("intent", "")
    if intent not in _VALID_INTENTS:
        logger.warning("Unrecognised intent '%s' - defaulting to faq.", intent)
        intent = "faq"

    confidence = parsed.get("confidence", 0.5)
    requires_tool = bool(parsed.get("requires_tool", intent in _TOOL_REQUIRING_INTENTS))

    logger.info("Classified intent=%s confidence=%.2f requires_tool=%s", intent, confidence, requires_tool)

    updates.update(
        {
            "intent": intent,
            "intent_confidence": confidence,
            "requires_tool": requires_tool,
        }
    )
    return updates
