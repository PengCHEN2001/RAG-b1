"""Extract evaluation inputs from a retained agent run trace.

The adapter is offline-only: it reads the terminal ``run.finished`` event and
does not resume the graph, call Gemini, execute SQL, or run either retriever.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent.runtime import read_trace


class TraceAdapterError(ValueError):
    """Raised when a trace cannot provide one complete evaluation record."""


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TraceAdapterError(f"run.finished {field} must be an object")
    return value


def _optional_mapping(value: Any, field: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _mapping(value, field)


def load_evaluation_trace(path: str | Path) -> dict[str, Any]:
    """Return the stable evaluation fields from one complete trace JSONL file."""
    trace_path = Path(path)
    trace = read_trace(trace_path)
    finished = [event for event in trace["events"] if event["event"] == "run.finished"]
    if not finished:
        details = "; ".join(trace["warnings"]) or "terminal event is missing"
        raise TraceAdapterError(f"Incomplete trace {trace_path}: {details}")
    if len(finished) != 1:  # read_trace currently enforces this; keep the adapter explicit.
        raise TraceAdapterError(f"Trace {trace_path} contains multiple terminal events")

    event = finished[0]
    payload = _mapping(event["payload"], "payload")
    state = _mapping(payload.get("state"), "payload.state")
    route_plan = _optional_mapping(state.get("route_plan"), "state.route_plan")
    sql_result = _optional_mapping(state.get("sql_result"), "state.sql_result")
    rulebook_result = _optional_mapping(
        state.get("rulebook_result"), "state.rulebook_result"
    )

    rulebook_results: list[dict[str, Any]] = []
    if rulebook_result is not None:
        raw_results = rulebook_result.get("results", [])
        if not isinstance(raw_results, list) or not all(
            isinstance(result, dict) for result in raw_results
        ):
            raise TraceAdapterError("state.rulebook_result.results must be a list of objects")
        rulebook_results = raw_results

    citations = state.get("citations", {})
    if not isinstance(citations, dict):
        raise TraceAdapterError("state.citations must be an object")
    step_durations = payload.get("step_durations_ms", {})
    if not isinstance(step_durations, dict) or not all(
        isinstance(name, str) and isinstance(duration, (int, float))
        for name, duration in step_durations.items()
    ):
        raise TraceAdapterError("step_durations_ms must map node names to numbers")

    duration_ms = payload.get("duration_ms")
    if not isinstance(duration_ms, (int, float)) or duration_ms < 0:
        raise TraceAdapterError("duration_ms must be a non-negative number")

    route = route_plan.get("route") if route_plan is not None else None
    if route is not None and not isinstance(route, str):
        raise TraceAdapterError("state.route_plan.route must be a string or null")

    return {
        "trace_path": str(trace_path),
        "session_id": event["session_id"],
        "run_id": event["run_id"],
        "question": state.get("user_input"),
        "standalone_question": state.get("standalone_question"),
        "route": route,
        "route_plan": route_plan,
        "sql_result": sql_result,
        "rulebook_results": rulebook_results,
        "evidence_status": state.get("evidence_status"),
        "answer": state.get("answer"),
        "citations": citations,
        "outcome": payload.get("outcome"),
        "duration_ms": duration_ms,
        "step_durations_ms": step_durations,
        "failed_node": payload.get("failed_node"),
        "error": payload.get("error"),
        # Legacy traces retain null; current traces carry observed SDK-call
        # counts, provider usage and explicit coverage without token estimates.
        "llm_calls": payload.get("llm_calls"),
        "usage": payload.get("usage"),
        "first_answer_token_ms": payload.get("first_answer_token_ms"),
        "trace_warnings": trace["warnings"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Path to one retained run JSONL file")
    args = parser.parse_args()
    print(json.dumps(load_evaluation_trace(args.trace), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
