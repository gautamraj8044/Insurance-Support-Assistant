"""
graph/state.py - LangGraph state schema for the insurance support workflow.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict


class GraphState(TypedDict, total=False):
    # -- Turn input -------------------------------------------------------
    user_message: str
    conversation_history: str   # Rendered "User: ... \nAssistant: ..." transcript

    # -- Intent classification ---------------------------------------------
    intent: Optional[str]              # e.g. "greeting", "faq", "policy_info"
    intent_confidence: Optional[float]
    requires_tool: Optional[bool]

    # -- Extracted entities (carried across turns by the caller) -----------
    policy_number: Optional[str]
    customer_id: Optional[str]
    claim_id: Optional[str]

    # -- Clarification flow ---------------------------------------------------
    needs_clarification: Optional[bool]

    # -- Output -------------------------------------------------------------
    final_answer: Optional[str]
    error: Optional[str]

    # -- Diagnostics / observability -----------------------------------------
    visited_nodes: List[str]
    trace_id: Optional[str]
