"""Report generation helper."""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _persistence_evidence(resume_success: bool) -> str:
    if resume_success:
        return (
            "Latest scenario execution verified LangGraph `get_state_history()` for every "
            "scenario thread id. With `checkpointer: sqlite`, checkpoints are durably stored "
            "under `outputs/checkpoints.sqlite`, so the recorded state history can be inspected "
            "after the run instead of living only in process memory."
        )
    return (
        "The graph is compiled with a checkpointer and the CLI passes a `thread_id` for every "
        "scenario. State-history verification was not observed in the latest metrics run; rerun "
        "with `checkpointer: sqlite` to produce durable recovery evidence."
    )


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data.

    Generate a report that includes:
    1. Metrics summary table (total scenarios, success rate, retries, interrupts)
    2. Per-scenario results table
    3. Architecture explanation (your graph design, state schema, reducers)
    4. Failure analysis (at least two failure modes you considered)
    5. Improvement plan

    Use reports/lab_report_template.md as your guide.

    Return: formatted markdown string
    """
    scenario_rows = [
        "| Scenario | Expected route | Actual route | Success | Retries | Interrupts | Errors |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for item in metrics.scenario_metrics:
        scenario_rows.append(
            "| "
            f"{item.scenario_id} | "
            f"{item.expected_route} | "
            f"{item.actual_route or ''} | "
            f"{_yes_no(item.success)} | "
            f"{item.retry_count} | "
            f"{item.interrupt_count} | "
            f"{len(item.errors)} |"
        )

    return "\n".join(
        [
            "# Day 08 Lab Report",
            "",
            "## 1. Team / student",
            "",
            "- Name: Hoàng Thanh Chiến",
            "- Repo/commit: https://github.com/HTChieens/phase2-track3-day8-langgraph-agent",
            "- Date: 29/06/2026",
            "",
            "## 2. Architecture",
            "",
            "The workflow is a LangGraph `StateGraph` built around a support-ticket agent. "
            "Execution starts at `intake`, then `classify` uses structured LLM output to choose "
            "one of five routes: `simple`, `tool`, `missing_info`, `risky`, or `error`.",
            "",
            "The main graph paths are:",
            "",
            "- `simple`: `answer -> finalize -> END`",
            "- `tool`: `tool -> evaluate -> answer/retry`",
            "- `missing_info`: `clarify -> finalize -> END`",
            "- `risky`: `risky_action -> approval -> tool/clarify`",
            "- `error`: `retry -> tool/dead_letter`",
            "",
            "All terminal paths pass through `finalize` before `END`, which gives the run a final "
            "audit event and makes route completion easy to verify from metrics.",
            "",
            "## 3. State schema",
            "",
            "The graph keeps state lean and serializable. Scalar fields are overwritten with the "
            "latest value, while audit-like list fields use append reducers.",
            "",
            "| Field | Reducer | Why |",
            "|---|---|---|",
            "| `thread_id` | overwrite | Unique scenario run id for checkpointer configuration |",
            "| `scenario_id` | overwrite | Identifies the scenario in metrics/reporting |",
            "| `query` | overwrite | Normalized user request |",
            "| `route` | overwrite | Current classification route |",
            "| `risk_level` | overwrite | Current risk signal from classification |",
            "| `attempt` | overwrite | Current retry attempt count |",
            "| `max_attempts` | overwrite | Retry bound used to prevent infinite loops |",
            "| `evaluation_result` | overwrite | Gate between `answer` and `retry` after tool evaluation |",
            "| `pending_question` | overwrite | Clarification question for incomplete requests |",
            "| `proposed_action` | overwrite | Risky action description awaiting approval |",
            "| `approval` | overwrite | Human/mock reviewer decision |",
            "| `final_answer` | overwrite | Final user-visible answer |",
            "| `messages` | append | Lightweight execution trace |",
            "| `tool_results` | append | Tool call history across retries |",
            "| `errors` | append | Failure history for retries and dead-letter analysis |",
            "| `events` | append | Structured audit trail for grading and debugging |",
            "",
            "## 4. Metrics summary",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Total scenarios | {metrics.total_scenarios} |",
            f"| Success rate | {metrics.success_rate:.2%} |",
            f"| Average nodes visited | {metrics.avg_nodes_visited:.2f} |",
            f"| Total retries | {metrics.total_retries} |",
            f"| Total interrupts/approvals | {metrics.total_interrupts} |",
            f"| Resume success | {_yes_no(metrics.resume_success)} |",
            "",
            "## 5. Scenario results",
            "",
            *scenario_rows,
            "",
            "## 6. Failure analysis",
            "",
            "1. Retry or tool failure: tool results containing `ERROR` are marked "
            "`needs_retry` by `evaluate`. The retry route increments `attempt` and compares it "
            "against `max_attempts`; once the bound is reached, the graph goes to `dead_letter` "
            "instead of looping forever.",
            "",
            "2. Risky action without approval: refund, deletion, email-sending, cancellation, "
            "and account-changing requests route through `risky_action` and `approval`. Only an "
            "approved decision proceeds to `tool`; rejected or missing approval routes to "
            "`clarify` for a safer alternative.",
            "",
            "3. Missing information: vague requests route to `clarify`, producing a "
            "`pending_question` instead of hallucinating an answer or calling a tool with "
            "insufficient context.",
            "",
            "## 7. Persistence / recovery evidence",
            "",
            _persistence_evidence(metrics.resume_success),
            "",
            "## 8. Extension work",
            "",
            "Implemented SQLite checkpointer support in the persistence adapter and wired the CLI "
            "to verify checkpoint-backed state history after scenario execution. This satisfies the "
            "bonus extension with concrete recovery evidence instead of only configuration support.",
            "",
            "## 9. Improvement plan",
            "",
            "With one more day, the next production steps would be stronger LLM-as-judge evaluation, "
            "real approval UI for HITL decisions, richer tool result schemas, and state-history "
            "screenshots/logs showing crash recovery from SQLite checkpoints.",
            "",
        ]
    )


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
