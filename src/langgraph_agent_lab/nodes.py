"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .state import AgentState, ApprovalDecision, Route, make_event


class ClassificationResult(BaseModel):
    """Structured LLM output for support-ticket routing."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description="Best route for the support request."
    )
    reasoning: str = Field(description="Brief reason for the selected route.")


def _load_env_file() -> None:
    """Load simple KEY=VALUE pairs from .env when the shell has not exported them."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _message_text(response: object) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return " ".join(str(item) for item in content).strip()
    return str(content).strip()


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── Workflow nodes ─────────────────────────────────────────────────


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    _load_env_file()
    from .llm import get_llm

    llm = get_llm(temperature=0.0).with_structured_output(ClassificationResult)
    query = state.get("query", "")
    result = llm.invoke(
        [
            (
                "system",
                "Classify a support-ticket request into exactly one route. "
                "Routes: risky, tool, missing_info, error, simple. "
                "Use this priority when more than one applies: "
                "risky > tool > missing_info > error > simple. "
                "risky means side effects such as refunds, deletions, sending emails, "
                "cancellations, account changes, or irreversible actions. "
                "tool means information lookup such as order status, tracking, or search. "
                "missing_info means the user request is too vague to act on. "
                "error means system failures such as timeout, crash, unavailable service. "
                "simple means answerable without tools or side effects.",
            ),
            ("human", f"Support request: {query}"),
        ]
    )
    route = result.route
    risk_level = "high" if route == Route.RISKY.value else "low"
    return {
        "route": route,
        "risk_level": risk_level,
        "messages": [f"classified:{route}"],
        "events": [
            make_event(
                "classify",
                "completed",
                "query classified",
                route=route,
                risk_level=risk_level,
                reasoning=result.reasoning,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    route = state.get("route", "")
    attempt = int(state.get("attempt", 0))
    query = state.get("query", "")
    proposed_action = state.get("proposed_action")

    if route == Route.ERROR.value and attempt < 2:
        result = f"ERROR: transient tool failure while processing request on attempt {attempt + 1}"
        event_type = "failed"
    elif route == Route.RISKY.value:
        result = f"Risky action completed after approval: {proposed_action or query}"
        event_type = "completed"
    elif route == Route.TOOL.value:
        result = f"Lookup result for request '{query}': mock record found and request can proceed."
        event_type = "completed"
    else:
        result = f"Tool processed request '{query}' successfully."
        event_type = "completed"

    return {
        "tool_results": [result],
        "messages": [f"tool:{event_type}"],
        "events": [make_event("tool", event_type, "mock tool executed", result=result)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    latest_result = (state.get("tool_results") or [""])[-1]
    evaluation_result = "needs_retry" if "ERROR" in latest_result.upper() else "success"
    return {
        "evaluation_result": evaluation_result,
        "messages": [f"evaluate:{evaluation_result}"],
        "events": [
            make_event(
                "evaluate",
                "completed",
                "tool result evaluated",
                evaluation_result=evaluation_result,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    _load_env_file()
    from .llm import get_llm

    llm = get_llm(temperature=0.2)
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")
    response = llm.invoke(
        [
            (
                "system",
                "You are a concise support agent. Answer using only the provided context. "
                "If tool results are present, ground the answer in them. "
                "If an approval decision is present, mention whether the approved action proceeded. "
                "Do not invent private customer details.",
            ),
            (
                "human",
                "Original request:\n"
                f"{query}\n\n"
                "Tool results:\n"
                f"{tool_results or 'None'}\n\n"
                "Approval decision:\n"
                f"{approval or 'None'}",
            ),
        ]
    )
    final_answer = _message_text(response)
    return {
        "final_answer": final_answer,
        "messages": ["answer:completed"],
        "events": [make_event("answer", "completed", "final answer generated")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    pending_question = (
        "Could you provide the specific account, order, issue details, and the action you want "
        f"taken for this request: '{query}'?"
    )
    return {
        "pending_question": pending_question,
        "final_answer": pending_question,
        "messages": ["clarify:pending"],
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    proposed_action = (
        f"Proposed risky support action: {query}. "
        "This requires approval because it may change customer data, trigger communication, "
        "or create a financial/account side effect."
    )
    return {
        "proposed_action": proposed_action,
        "messages": ["risky_action:prepared"],
        "events": [make_event("risky_action", "completed", "risky action prepared")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return: {"approval": {"approved": bool, "reviewer": str, "comment": str}, "events": [make_event(...)]}
    """
    _load_env_file()
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        payload = interrupt(
            {
                "proposed_action": state.get("proposed_action"),
                "question": "Approve this risky support action?",
            }
        )
        decision = ApprovalDecision.model_validate(payload).model_dump()
    else:
        decision = ApprovalDecision(
            approved=True,
            reviewer="mock-reviewer",
            comment="Approved by default mock reviewer for lab execution.",
        ).model_dump()

    return {
        "approval": decision,
        "messages": [f"approval:{'approved' if decision['approved'] else 'rejected'}"],
        "events": [
            make_event(
                "approval",
                "completed",
                "approval decision recorded",
                approved=decision["approved"],
                reviewer=decision["reviewer"],
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    current_attempt = int(state.get("attempt", 0))
    next_attempt = current_attempt + 1
    latest_result = (state.get("tool_results") or ["no tool result available"])[-1]
    error = f"Attempt {next_attempt} scheduled after failure: {latest_result}"
    return {
        "attempt": next_attempt,
        "errors": [error],
        "messages": [f"retry:{next_attempt}"],
        "events": [
            make_event(
                "retry",
                "completed",
                "retry attempt recorded",
                attempt=next_attempt,
                max_attempts=state.get("max_attempts", 3),
            )
        ],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    final_answer = (
        "I could not complete this request after the allowed retry attempts. "
        "The issue has been moved to manual review with the recorded error history."
    )
    return {
        "final_answer": final_answer,
        "route": Route.ERROR.value,
        "messages": ["dead_letter:completed"],
        "events": [make_event("dead_letter", "completed", "max retries exhausted")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {
        "messages": ["finalize:completed"],
        "events": [make_event("finalize", "completed", "workflow finished")],
    }
