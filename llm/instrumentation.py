"""Observe SDK calls and aggregate provider usage without inventing missing data."""

from time import perf_counter
from uuid import uuid4


TOKEN_FIELDS = (
    "prompt_token_count", "candidates_token_count", "thoughts_token_count",
    "cached_content_token_count", "tool_use_prompt_token_count", "total_token_count",
)


def generate_content(client, *, purpose, on_event=None, cancel_event=None, **kwargs):
    """Return the SDK response unchanged; record usage before caller validation.

    One call means one SDK invocation, including errors, not internal HTTP
    attempts. Observers receive metadata only, never prompts/provider errors.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise KeyboardInterrupt
    started = perf_counter()
    call = {"call_id": uuid4().hex, "purpose": purpose, "provider": "google",
            "model": kwargs["model"], "usage": None}

    def emit(kind, **update):
        if on_event is not None:
            on_event({"event": kind, "payload": {**call, **update}})

    emit("llm.started")
    try:
        response = client.models.generate_content(**kwargs)
    except (Exception, KeyboardInterrupt) as error:
        emit("llm.failed", duration_ms=(perf_counter() - started) * 1000,
             error={"type": type(error).__name__})
        raise
    metadata = getattr(response, "usage_metadata", None)
    usage = None
    if metadata is not None:
        # SDK fields are optional. Reject non-numeric telemetry at this external
        # boundary without letting bad telemetry discard an otherwise valid reply.
        usage = {field: value if type(value := getattr(metadata, field, None)) is int and value >= 0 else None
                 for field in TOKEN_FIELDS}
    details = {field: value if isinstance(value := getattr(response, field, None), str) else None
               for field in ("response_id", "model_version")}
    emit("llm.finished", duration_ms=(perf_counter() - started) * 1000, usage=usage, **details)
    # An in-flight response still contributes usage when its answer is cancelled.
    if cancel_event is not None and cancel_event.is_set():
        raise KeyboardInterrupt
    return response


def summarize_usage(usages, *, instrumentation_complete=True):
    """Unknown calls contribute null, with reported subtotals kept separately."""
    return merge_usage([
        {"total_calls": 1, "instrumentation_complete": instrumentation_complete,
         "known_totals": usage or dict.fromkeys(TOKEN_FIELDS),
         "reported_calls": {field: int(usage is not None and usage[field] is not None) for field in TOKEN_FIELDS}}
        for usage in usages
    ], instrumentation_complete=instrumentation_complete)


def merge_usage(summaries, *, instrumentation_complete=True):
    """Combine run summaries; legacy null summaries leave session totals unknown."""
    summaries = list(summaries)
    covered = instrumentation_complete and all(
        item is not None and item["instrumentation_complete"] for item in summaries)
    available = [item for item in summaries if item is not None]
    count = sum(item["total_calls"] for item in available)
    known, reported = {}, {}
    for field in TOKEN_FIELDS:
        values = [item["known_totals"][field] for item in available if item["known_totals"][field] is not None]
        known[field] = sum(values) if values else None
        reported[field] = sum(item["reported_calls"][field] for item in available)
    return {
        **{field: known[field] if covered and count and reported[field] == count else None for field in TOKEN_FIELDS},
        "known_totals": known, "reported_calls": reported, "total_calls": count,
        "instrumentation_complete": covered,
        "complete": covered and count > 0 and all(reported[field] == count for field in (
            "prompt_token_count", "candidates_token_count", "total_token_count")),
    }


def session_stats(runs):
    """Callers key runs by run_id so offline selection cannot double-count them."""
    runs = list(runs)
    calls = [run.get("llm_calls") for run in runs]
    return {"llm_calls": sum(calls) if all(value is not None for value in calls) else None,
            "usage": merge_usage(run.get("usage") for run in runs)}


def format_usage(usage):
    if usage is None:
        return "usage unavailable"
    parts = [f"{label}={usage[field] if usage[field] is not None else 'null'}" for label, field in (
        ("in", "prompt_token_count"), ("out", "candidates_token_count"), ("total", "total_token_count"))]
    parts.append(f"reported={usage['reported_calls']['total_token_count']}/{usage['total_calls']}")
    if usage["total_token_count"] is None and usage["known_totals"]["total_token_count"] is not None:
        parts.append(f"known_total={usage['known_totals']['total_token_count']}")
    if not usage["instrumentation_complete"]:
        parts.append("scope=partial")
    return " ".join(parts)
