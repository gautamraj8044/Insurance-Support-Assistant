"""
ui/app.py - Streamlit chat interface for the Insurance Support Assistant.

Run with:
    streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running `streamlit run ui/app.py` from the project root by adding
# the project root to sys.path (Streamlit doesn't treat ui/ as a package).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from core.conversation import ConversationSession, run_turn
from core.logging_setup import configure_logging, get_logger
from core.settings import app as app_settings
from core.settings import groq, validate_settings
from graph.builder import build_app
from tools.faq_tool import is_faq_store_ready

configure_logging()
logger = get_logger(__name__)


st.set_page_config(
    page_title=app_settings.app_title,
    page_icon=":shield:",
    layout="centered",
)


# -- Cached resources -----------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_graph_app():
    return build_app()


@st.cache_resource(show_spinner=False)
def _check_faq_store() -> bool:
    return is_faq_store_ready()


def _get_session() -> ConversationSession:
    if "session" not in st.session_state:
        st.session_state.session = ConversationSession()
    return st.session_state.session


# -- Sidebar ----------------------------------------------------------------------

def _render_sidebar(config_problems: list[str], faq_ready: bool) -> None:
    with st.sidebar:
        st.title(app_settings.app_title)
        st.caption("Insurance customer support, powered by an LLM agent.")

        st.divider()
        st.subheader("System status")

        if config_problems:
            for p in config_problems:
                st.error(p, icon=":material/error:")
        else:
            st.success("Groq API key configured", icon=":material/check_circle:")

        if faq_ready:
            st.success("Knowledge base loaded", icon=":material/check_circle:")
        else:
            st.warning(
                "Knowledge base not loaded. Run `python setup.py` first.",
                icon=":material/warning:",
            )

        st.divider()
        st.subheader("Model")
        st.text(f"Provider: Groq")
        st.text(f"Model: {groq.model}")
        st.text(f"Intent model: {groq.intent_model}")

        session = _get_session()
        if session.policy_number or session.customer_id or session.claim_id:
            st.divider()
            st.subheader("Session context")
            if session.policy_number:
                st.text(f"Policy: {session.policy_number}")
            if session.customer_id:
                st.text(f"Customer: {session.customer_id}")
            if session.claim_id:
                st.text(f"Claim: {session.claim_id}")

        st.divider()
        if st.button("Reset conversation", use_container_width=True):
            session.reset()
            st.session_state.pop("messages_rendered", None)
            st.rerun()


# -- Main chat area -----------------------------------------------------------------

def _render_chat_history(session: ConversationSession) -> None:
    for turn in session.turns:
        with st.chat_message(turn.role):
            st.markdown(turn.content)


def main() -> None:
    config_problems = validate_settings()
    faq_ready = _check_faq_store()

    _render_sidebar(config_problems, faq_ready)

    st.title(":shield: Insurance Support Assistant")
    st.caption("Ask about policies, claims, premiums, renewals, or general coverage questions.")

    if config_problems:
        st.error(
            "The app is not fully configured yet. Please set GROQ_API_KEY in your "
            "environment or .env file before chatting.",
        )
        st.stop()

    session = _get_session()
    _render_chat_history(session)

    user_message = st.chat_input("Type your question here...")
    if not user_message:
        return

    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                graph_app = _load_graph_app()
                answer = run_turn(session, user_message, app=graph_app)
            except Exception:
                logger.exception("Unexpected error in Streamlit turn handler")
                answer = (
                    "Sorry, something went wrong on our end. Please try again "
                    "in a moment."
                )
        st.markdown(answer)


if __name__ == "__main__":
    main()
