"""Run one graph turn and retain a process-independent, graph-level trace."""

import hashlib
import json
import os
import platform
import traceback
from dataclasses import fields
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.types import Overwrite
from pydantic import ValidationError

from retrieval.rulebook_retriever import RetrievedChunk, RetrievedRulebookContext, RulebookQuerySet
from sql_retriever import RetrievalResult


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACE_DIR = ROOT / "data" / "traces"
_DOMAIN_TYPES = (RulebookQuerySet, RetrievedChunk, RetrievedRulebookContext, RetrievalResult)
_SECRET_FIELDS = {
    "api_key", "api-key", "authorization", "proxy-authorization", "x-api-key",
    "x-goog-api-key", "access_token", "refresh_token", "client_secret",
}


def trace_value(value, secrets=()):
    """Snapshot known state types; never serialize SDK clients or object reprs."""
    if isinstance(value, BaseMessage):
        value = {"type": value.type, "id": value.id, "content": value.content}
    elif isinstance(value, Overwrite):
        value = {"$overwrite": value.value}
    elif isinstance(value, _DOMAIN_TYPES):
        value = {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, dict):
        return {key: "[REDACTED]" if key.lower() in _SECRET_FIELDS else trace_value(item, secrets)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [trace_value(item, secrets) for item in value]
    if isinstance(value, bytes):
        return {"$bytes_hex": value.hex()}
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError(f"Unsupported trace value: {type(value).__name__}")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _code_fingerprint():
    digest = hashlib.sha256()
    for directory in ("agent", "ui", "retrieval", "sql_retriever", "ingestion", "llm"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            digest.update(str(path.relative_to(ROOT)).encode() + b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


class _TraceFile:
    """IO failures disable the writer without losing a valid interaction result."""

    def __init__(self, root, session_id, run_id, model):
        self.path = Path(root) / session_id / f"{run_id}.jsonl" if root is not None else None
        self.complete = True if self.path is not None else None
        self.failure = None
        self.file = None
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            metadata = self.path.parent / "session.json"
            if not metadata.exists():
                metadata.write_text(json.dumps({
                    "schema_version": 1, "session_id": session_id, "created_at": _now(),
                    "python": platform.python_version(), "langgraph": version("langgraph"),
                    "initial_model": model,
                    "uv_lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
                    "code_sha256": _code_fingerprint(), "code_snapshot": None, "data_versions": None,
                    "trace_scope": "graph", "llm_requests": None,
                    "retriever_internals": None, "token_usage": None,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            self.file = self.path.open("x", encoding="utf-8")
        except OSError as error:
            self._fail(error)

    def _fail(self, error):
        self.complete = False
        self.failure = type(error).__name__
        self.close()

    def write(self, event):
        if self.file is None:
            return
        try:
            self.file.write(json.dumps(event, ensure_ascii=False) + "\n")
            self.file.flush()
        except (OSError, TypeError, ValueError) as error:
            self._fail(error)

    def close(self):
        handle, self.file = self.file, None
        if handle is not None:
            try:
                handle.close()
            except OSError as error:
                self.complete = False
                self.failure = self.failure or type(error).__name__


def _exception_info(error):
    # Provider reprs and validation input dumps can contain credentials. Keep
    # exception type, validation locations and stack locations, without locals,
    # source lines or raw provider messages.
    info = {
        "type": type(error).__name__,
        "message": "Model output failed validation." if isinstance(error, ValidationError) else "Graph execution failed.",
        "stack": [{"file": frame.filename, "line": frame.lineno, "function": frame.name}
                  for frame in traceback.extract_tb(error.__traceback__)],
    }
    if isinstance(error, ValidationError):
        info["validation_errors"] = [{"location": list(item["loc"]), "type": item["type"]}
                                     for item in error.errors(include_input=False, include_context=False)]
    return info


def run_turn(graph, context, *, session_id, question, trace_dir=DEFAULT_TRACE_DIR, on_event=None):
    """Run synchronously; caller-generated session IDs are internal identifiers.

    The callback receives the same redacted events as disk. --trace display and
    persistence are independent. Errors/cancellation return a terminal result;
    no failed node is retried. Only the graph's completed turns enter its memory.
    """
    started = perf_counter()
    run_id = uuid4().hex
    config = {"configurable": {"thread_id": session_id}}
    previous = graph.get_state(config).values
    writer = _TraceFile(trace_dir, session_id, run_id, context.model)
    seq, warned = 0, False
    active, durations = {}, {}
    failed_node = None
    error = None
    prepared = False
    state = previous
    outcome = "error"

    def emit(event, payload, *, node=None, span_id=None):
        nonlocal seq, warned
        seq += 1
        secrets = [os.getenv(name) for name in ("LLM_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY")]
        secrets.append(getattr(context.client, "api_key", None))
        # ponytail: inline node snapshots; use shared content-addressed blobs if
        # measured trace size makes repeated evidence snapshots expensive.
        record = trace_value({
            "schema_version": 1, "session_id": session_id, "run_id": run_id,
            "seq": seq, "timestamp": _now(), "event": event, "node": node,
            "span_id": span_id or run_id, "parent_span_id": run_id if span_id else None,
            "payload": payload,
        }, [secret for secret in secrets if isinstance(secret, str) and secret])
        writer.write(record)
        if on_event is not None:
            on_event(record)
        if writer.failure is not None and not warned:
            warned = True
            emit("trace.failed", {"error_type": writer.failure, "message": "Trace is incomplete; interaction can continue."})

    emit("run.started", {
        "user_input": question, "turn_index": previous.get("turn_index", 0) + 1,
        "model": context.model, "previous_messages": previous.get("messages", []),
        "conversation_context": previous.get("conversation_context", {}),
        "pending_clarification": previous.get("pending_clarification"),
        "trace_enabled": trace_dir is not None, "llm_calls": None, "usage": None,
    })
    try:
        for task in graph.stream({"user_input": question, "run_id": run_id}, config,
                                 context=context, stream_mode="tasks"):
            task_id, node = task["id"], task["name"]
            if "input" in task:
                active[task_id] = (node, perf_counter())
                emit("step.started", {"input": task["input"]}, node=node, span_id=task_id)
                continue
            _, step_started = active.pop(task_id)
            duration = (perf_counter() - step_started) * 1000
            durations[node] = duration
            if task["error"] is not None:
                failed_node = node
                emit("step.failed", {"duration_ms": duration, "error": "Node failed; see run.finished error details."},
                     node=node, span_id=task_id)
                continue
            update = task["result"]
            prepared = prepared or node == "prepare_turn"
            emit("step.finished", {"duration_ms": duration, "update": update}, node=node, span_id=task_id)
            if node == "plan" or node == "prepare_rulebook_query":
                emit("route.selected" if node == "plan" else "query.refined", update, node=node, span_id=task_id)
            if node in ("retrieve_sql", "retrieve_rulebook"):
                if node == "retrieve_sql":
                    result = update["sql_result"]
                    summary = {"source": "sql", "status": result.status, "strategy": result.strategy,
                               "count": len(result.rows), "truncated": result.truncated, "error_code": result.error_code}
                else:
                    result = update.get("rulebook_result")
                    summary = {"source": "rulebook", "status": "error" if result is None else "ok" if result.results else "empty",
                               "count": len(result.results) if result is not None else 0,
                               "errors": update.get("retrieval_errors", {})}
                emit("retrieval.completed", summary, node=node, span_id=task_id)
        state = graph.get_state(config).values
        outcome = state["outcome"]
    except (Exception, KeyboardInterrupt) as failure:
        outcome = "cancelled" if isinstance(failure, KeyboardInterrupt) else "error"
        error = _exception_info(failure)
        for task_id, (node, step_started) in active.items():
            failed_node = node
            duration = (perf_counter() - step_started) * 1000
            durations[node] = duration
            emit("step.failed", {"duration_ms": duration, "error": error}, node=node, span_id=task_id)
        state = dict(graph.get_state(config).values)
        state.update(answer=None, citations={}, outcome=outcome, context_updates={})
        if not prepared:
            state.update(route_plan=None, sql_result=None, rulebook_result=None)
    finally:
        duration = (perf_counter() - started) * 1000
        # finish is emitted only here, including graph errors and Ctrl-C. A killed
        # process may leave only a prefix; the offline reader calls it interrupted.
        emit("run.finished", {
            "outcome": outcome, "duration_ms": duration, "step_durations_ms": durations,
            "failed_node": failed_node, "error": error, "state": state,
            "llm_calls": None, "usage": None, "first_answer_token_ms": None,
        })
        writer.close()
        if writer.failure is not None and not warned:
            emit("trace.failed", {"error_type": writer.failure, "message": "Trace is incomplete; interaction can continue."})
    return {
        "session_id": session_id, "run_id": run_id, "state": state, "outcome": outcome,
        "duration_ms": duration, "step_durations_ms": durations, "error": error,
        "failed_node": failed_node, "trace_path": writer.path, "trace_complete": writer.complete,
    }


def read_trace(path):
    """Read one run without importing clients, executing SQL or resuming a graph."""
    events, warnings = [], []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                if not line.endswith("\n"):
                    warnings.append(f"Skipped incomplete final line {number}.")
                    break
                raise ValueError(f"Invalid JSON on trace line {number}.") from error
            if (not isinstance(event, dict) or event.get("schema_version") != 1
                    or not {"session_id", "run_id", "seq", "event", "payload"}.issubset(event)
                    or not isinstance(event["payload"], dict) or not isinstance(event["event"], str)):
                raise ValueError(f"Invalid trace schema on line {number}.")
            if event["seq"] != number or (events and any(event[key] != events[0][key] for key in ("session_id", "run_id"))):
                raise ValueError(f"Mixed runs or invalid sequence on trace line {number}.")
            events.append(event)
    finished = [event for event in events if event["event"] == "run.finished"]
    if len(finished) > 1:
        raise ValueError("Trace contains multiple run.finished events.")
    return {"events": events, "warnings": warnings,
            "outcome": finished[0]["payload"]["outcome"] if finished else "interrupted"}


def source_details(state):
    """Resolve only this answer's cited sources, also accepting offline JSON state."""
    state = trace_value(state)
    result = {}
    for ref, citation in state.get("citations", {}).items():
        if citation["source"] == "sql":
            result[ref] = {"query_id": citation["source_id"], **state["sql_result"],
                           "answer_context_truncated": state.get("answer_context_truncated", {}).get("sql", False),
                           "query_context_truncated": state.get("sql_context_truncated", False)}
        else:
            chunk = next(item for item in state["rulebook_result"]["results"]
                         if item["chunk"]["chunk_id"] == citation["source_id"])
            result[ref] = {**chunk, "answer_context_truncated": state.get("answer_context_truncated", {}).get("rulebook", False)}
    return result
