"""
graph/nodes/database_agent.py - Account-specific lookups via tool calling.

Consolidates policy/billing/claims/renewal lookups into a single node.
The LLM decides which tool(s) to call based on the user's request and
the intent classified upstream; if required info (policy number, etc.)
is missing, it asks for it instead of guessing.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from core.llm_client import get_llm, safe_invoke
from core.logging_setup import get_logger
from core.tracing import timed_node
from graph.state import GraphState
from prompts.templates import DATABASE_AGENT_PROMPT
from tools.db_tools import (
    get_auto_policy_details,
    get_billing_info,
    get_claim_status,
    get_payment_history,
    get_policy_details,
    get_renewal_status,
)

logger = get_logger(__name__)

_FALLBACK_TEXT = (
    "I'm having trouble retrieving your account details right now. "
    "Please try again shortly, or I can connect you with a representative."
)

# -- Tool schema definitions (OpenAI/Groq-compatible function schemas) --------

_TOOL_SPECS = [
    {
        "name": "get_policy_details",
        "description": "Fetch general policy info (type, premium, status) by policy number.",
        "params": {"policy_number": ("string", True, "The policy number, e.g. POL000123.")},
        "func": get_policy_details,
    },
    {
        "name": "get_auto_policy_details",
        "description": "Fetch auto-specific policy details (vehicle, deductibles) by policy number.",
        "params": {"policy_number": ("string", True, "The policy number, e.g. POL000123.")},
        "func": get_auto_policy_details,
    },
    {
        "name": "get_claim_status",
        "description": "Fetch claim details by claim ID, or the latest claims for a policy number.",
        "params": {
            "claim_id": ("string", False, "The claim ID, e.g. CLM000123."),
            "policy_number": ("string", False, "The policy number, e.g. POL000123."),
        },
        "func": get_claim_status,
    },
    {
        "name": "get_billing_info",
        "description": "Fetch the most recent pending billing record for a policy or customer.",
        "params": {
            "policy_number": ("string", False, "The policy number, e.g. POL000123."),
            "customer_id": ("string", False, "The customer ID, e.g. CUST00123."),
        },
        "func": get_billing_info,
    },
    {
        "name": "get_payment_history",
        "description": "Fetch the last 10 payments for a policy.",
        "params": {"policy_number": ("string", True, "The policy number, e.g. POL000123.")},
        "func": get_payment_history,
    },
    {
        "name": "get_renewal_status",
        "description": "Fetch renewal-relevant info (status, start date) for a policy.",
        "params": {"policy_number": ("string", True, "The policy number, e.g. POL000123.")},
        "func": get_renewal_status,
    },
]


def _build_tools() -> list[StructuredTool]:
    tools = []
    for spec in _TOOL_SPECS:
        fields = {}
        for pname, (ptype, required, pdesc) in spec["params"].items():
            py_type = str if ptype == "string" else object
            fields[pname] = (
                py_type,
                Field(default=... if required else None, description=pdesc),
            )
        args_model = create_model(f"{spec['name']}_args", **fields)
        tools.append(
            StructuredTool(
                name=spec["name"],
                description=spec["description"],
                args_schema=args_model,
                func=spec["func"],
            )
        )
    return tools


_TOOLS = _build_tools()
_TOOL_MAP = {t.name: t for t in _TOOLS}


@timed_node("database_agent")
def database_agent_node(state: GraphState) -> GraphState:
    user_message = state.get("user_message", "")
    conversation_history = state.get("conversation_history", "")

    prompt = DATABASE_AGENT_PROMPT.format(
        user_message=user_message,
        intent=state.get("intent", "unknown"),
        conversation_history=conversation_history or "(no prior messages)",
        policy_number=state.get("policy_number") or "Not provided",
        customer_id=state.get("customer_id") or "Not provided",
        claim_id=state.get("claim_id") or "Not provided",
    )

    llm = get_llm().bind_tools(_TOOLS)
    messages = [HumanMessage(content=prompt)]

    first = safe_invoke(llm, messages, fallback_text=_FALLBACK_TEXT)
    if not first.succeeded:
        return {"final_answer": first.text, "error": first.error}

    if not first.tool_calls:
        # LLM answered directly (e.g. asked a clarifying question) - no tool needed.
        return {"final_answer": first.text}

    # Execute each requested tool call.
    tool_results = []
    for tc in first.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_id = tc["id"]
        tool = _TOOL_MAP.get(tool_name)

        logger.info("Database agent invoking tool '%s' with args=%s", tool_name, tool_args)
        try:
            output = tool.invoke(tool_args) if tool else {"error": f"Unknown tool '{tool_name}'"}
        except Exception as exc:
            logger.exception("Tool '%s' raised an error", tool_name)
            output = {"error": str(exc)}

        tool_results.append(ToolMessage(content=json.dumps(output), tool_call_id=tool_id))

    # Second call: let the LLM turn raw tool output into a friendly answer.
    followup_messages = [*messages, first_ai_message(first), *tool_results]
    final = safe_invoke(llm, followup_messages, fallback_text=_FALLBACK_TEXT)
    return {"final_answer": final.text, "error": final.error if not final.succeeded else None}


def first_ai_message(result):
    """
    Reconstruct a minimal AIMessage-like object carrying the original
    tool_calls, required by the LangChain message history when replaying
    tool results back to the model.
    """
    from langchain_core.messages import AIMessage

    return AIMessage(content=result.text or "", tool_calls=result.tool_calls)
