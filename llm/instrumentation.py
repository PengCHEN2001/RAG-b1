"""Observe, pace and retry Gemini SDK calls without inventing usage data."""

import os
from threading import Lock
from time import monotonic, perf_counter, sleep
from uuid import uuid4


TOKEN_FIELDS = (
    "prompt_token_count", "candidates_token_count", "thoughts_token_count",
    "cached_content_token_count", "tool_use_prompt_token_count", "total_token_count",
)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _env_number(name, default, cast):
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = cast(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


class _GeminiThrottle:
    """Reserve process-wide call slots so graph nodes share one rate limit."""

    def __init__(self):
        self._lock = Lock()
        self._next_call_at = 0.0

    def wait(self, interval, *, cancel_event=None, on_wait=None):
        if interval <= 0:
            return
        with self._lock:
            delay = max(0.0, self._next_call_at - monotonic())
            if delay > 0 and on_wait is not None:
                on_wait(delay)
            if delay > 0:
                if cancel_event is not None:
                    if cancel_event.wait(delay):
                        raise KeyboardInterrupt
                else:
                    sleep(delay)
            self._next_call_at = monotonic() + interval


_GEMINI_THROTTLE = _GeminiThrottle()


def _google_sdk_client(client):
    """Keep injected offline/mock clients fast while pacing the real SDK."""
    return type(client).__module__.startswith("google.genai")


def _status_code(error):
    for value in (
        getattr(error, "code", None),
        getattr(error, "status_code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        if isinstance(value, int):
            return value
    return None


def _retryable_provider_error(error):
    code = _status_code(error)
    return code in _RETRYABLE_STATUS_CODES or type(error).__name__ == "ServerError"


def _wait(delay, cancel_event=None):
    if delay <= 0:
        return
    if cancel_event is not None:
        if cancel_event.wait(delay):
            raise KeyboardInterrupt
    else:
        sleep(delay)


def generate_content(
    client, *, purpose, on_event=None, cancel_event=None,
    provider_controls=None, **kwargs,
):
    """Return the SDK response unchanged; record usage before caller validation.

    One call means one SDK invocation, including every retry attempt. Observers
    receive metadata only, never prompts or raw provider errors.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise KeyboardInterrupt
    controlled = _google_sdk_client(client) if provider_controls is None else provider_controls
    interval = _env_number("GEMINI_MIN_INTERVAL_SECONDS", 5.0, float) if controlled else 0.0
    max_attempts = max(1, _env_number("GEMINI_MAX_ATTEMPTS", 3, int)) if controlled else 1
    retry_base = _env_number("GEMINI_RETRY_BASE_SECONDS", 5.0, float)

    for attempt in range(1, max_attempts + 1):
        call = {
            "call_id": uuid4().hex, "purpose": purpose, "provider": "google",
            "model": kwargs["model"], "usage": None,
            "attempt": attempt, "max_attempts": max_attempts,
        }

        def emit(kind, **update):
            if on_event is not None:
                on_event({"event": kind, "payload": {**call, **update}})

        _GEMINI_THROTTLE.wait(
            interval, cancel_event=cancel_event,
            on_wait=lambda delay: emit("llm.waiting", wait_seconds=delay,
                                       reason="provider_rate_limit"),
        )
        started = perf_counter()
        emit("llm.started")
        try:
            response = client.models.generate_content(**kwargs)
        except (Exception, KeyboardInterrupt) as error:
            duration = (perf_counter() - started) * 1000
            emit("llm.failed", duration_ms=duration, error={"type": type(error).__name__})
            if (isinstance(error, KeyboardInterrupt) or attempt >= max_attempts
                    or not _retryable_provider_error(error)):
                raise
            delay = retry_base * (2 ** (attempt - 1))
            emit("llm.retrying", retry_in_seconds=delay,
                 error={"type": type(error).__name__})
            _wait(delay, cancel_event)
            continue
        metadata = getattr(response, "usage_metadata", None)
        usage = None
        if metadata is not None:
            # SDK fields are optional. Reject non-numeric telemetry at this external
            # boundary without letting bad telemetry discard an otherwise valid reply.
            usage = {
                field: value if type(value := getattr(metadata, field, None)) is int and value >= 0 else None
                for field in TOKEN_FIELDS
            }
        details = {
            field: value if isinstance(value := getattr(response, field, None), str) else None
            for field in ("response_id", "model_version")
        }
        emit("llm.finished", duration_ms=(perf_counter() - started) * 1000,
             usage=usage, **details)
        # An in-flight response still contributes usage when its answer is cancelled.
        if cancel_event is not None and cancel_event.is_set():
            raise KeyboardInterrupt
        return response
    raise AssertionError("Gemini retry loop exhausted without returning or raising")


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
