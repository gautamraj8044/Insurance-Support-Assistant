"""
core/conversation.py - Orchestrates a single user turn through the graph.

This is the only module the Streamlit UI talks to. It owns:
  - Rendering conversation history into the flat string format the
    prompts expect.
  - Persisting extracted entities (policy_number, customer_id, claim_id)
    across turns within a session.
  - Assigning a trace id to each turn for log correlation.
  - Translating any unexpected exception into a safe, user-facing message
    (the Streamlit layer should never see a raw traceback).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from core.logging_setup import get_logger
from core.tracing import new_trace_id, set_trace_id
from graph.builder import build_app

logger = get_logger(__name__)


@dataclass
class Turn:
    role: str   # "user" | "assistant"
    content: str


@dataclass
class ConversationSession:
    """
    Holds everything that needs to persist across turns for one chat
    session. Streamlit keeps one of these in st.session_state.
    """

    turns: List[Turn] = field(default_factory=list)
    policy_number: Optional[str] = None
    customer_id: Optional[str] = None
    claim_id: Optional[str] = None
    last_intent: Optional[str] = None
    last_error: Optional[str] = None

    def render_history(self, max_turns: int = 12) -> str:
        """Render the last N turns as a flat transcript for prompts."""
        recent = self.turns[-max_turns:]
        lines = [f"{t.role.capitalize()}: {t.content}" for t in recent]
        return "\n".join(lines)

    def add_user_turn(self, content: str) -> None:
        self.turns.append(Turn(role="user", content=content))

    def add_assistant_turn(self, content: str) -> None:
        self.turns.append(Turn(role="assistant", content=content))

    def reset(self) -> None:
        self.turns.clear()
        self.policy_number = None
        self.customer_id = None
        self.claim_id = None
        self.last_intent = None
        self.last_error = None


_GENERIC_FAILURE_MESSAGE = (
    "Sorry, something went wrong while processing your request. "
    "Please try again in a moment."
)


def run_turn(session: ConversationSession, user_message: str, app=None) -> str:
    """
    Run one user message through the graph and update the session
    in-place with any newly extracted entities and the new turns.

    Returns the assistant's reply text. Never raises - any failure is
    converted into a safe fallback message.
    """
    trace_id = new_trace_id()
    set_trace_id(trace_id)
    logger.info("Starting turn (trace_id=%s)", trace_id)

    session.add_user_turn(user_message)

    initial_state = {
        "user_message": user_message,
        "conversation_history": "\n".join(
            f"{t.role.capitalize()}: {t.content}" for t in session.turns[:-1][-12:]
        ),
        "policy_number": session.policy_number,
        "customer_id": session.customer_id,
        "claim_id": session.claim_id,
        "visited_nodes": [],
        "trace_id": trace_id,
    }

    try:
        graph_app = app or build_app()
        final_state = graph_app.invoke(initial_state)
    except Exception:
        logger.exception("Unhandled error running graph turn (trace_id=%s)", trace_id)
        session.add_assistant_turn(_GENERIC_FAILURE_MESSAGE)
        session.last_error = "internal_error"
        return _GENERIC_FAILURE_MESSAGE

    # Persist any newly discovered entities for future turns.
    session.policy_number = final_state.get("policy_number") or session.policy_number
    session.customer_id = final_state.get("customer_id") or session.customer_id
    session.claim_id = final_state.get("claim_id") or session.claim_id
    session.last_intent = final_state.get("intent")
    session.last_error = final_state.get("error")

    answer = final_state.get("final_answer") or _GENERIC_FAILURE_MESSAGE
    session.add_assistant_turn(answer)

    logger.info(
        "Turn complete (trace_id=%s, intent=%s, error=%s)",
        trace_id,
        session.last_intent,
        bool(session.last_error),
    )
    return answer
