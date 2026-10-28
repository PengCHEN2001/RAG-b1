"""Offline, deterministic metrics for Rulebook and Text-to-SQL results.

These functions do not call an LLM, RAGAS, a retriever, or SQLite. They compare
one frozen benchmark case with one already-captured system result. RAGAS judge
metrics and router classification metrics belong to later evaluation stages.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping, Sequence


class MetricInputError(ValueError):
    """Raised when a benchmark case or captured result cannot be scored."""


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise MetricInputError("At least one score is required for aggregation")
    return sum(values) / len(values)


def _retrieved_chunk_id(result: Any, position: int) -> str:
    if not isinstance(result, Mapping):
        raise MetricInputError(f"Retrieved result {position} must be an object")
    chunk = result.get("chunk")
    chunk_id = chunk.get("chunk_id") if isinstance(chunk, Mapping) else result.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise MetricInputError(f"Retrieved result {position} has no chunk_id")
    return chunk_id.strip()


def score_rulebook_case(
    case: Mapping[str, Any],
    retrieved_results: Sequence[Mapping[str, Any]],
    *,
    k: int = 5,
) -> dict[str, Any]:
    """Score one ranked Rulebook retrieval result using reference chunk IDs."""
    if k < 1:
        raise MetricInputError("k must be positive")
    case_id = case.get("case_id")
    contexts = case.get("reference_contexts")
    if not isinstance(case_id, str) or not isinstance(contexts, list) or not contexts:
        raise MetricInputError("Rulebook case requires case_id and reference_contexts")

    reference_ids: set[str] = set()
    for context in contexts:
        if not isinstance(context, Mapping):
            raise MetricInputError(f"{case_id}: reference context must be an object")
        chunk_id = context.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise MetricInputError(f"{case_id}: reference context has no chunk_id")
        reference_ids.add(chunk_id.strip())

    retrieved_ids = [
        _retrieved_chunk_id(result, position)
        for position, result in enumerate(retrieved_results, 1)
    ]
    relevant_at_k = reference_ids.intersection(retrieved_ids[:k])
    first_relevant_rank = next(
        (rank for rank, chunk_id in enumerate(retrieved_ids, 1) if chunk_id in reference_ids),
        None,
    )

    return {
        "case_id": case_id,
        "k": k,
        "reference_chunk_count": len(reference_ids),
        "retrieved_chunk_count": len(retrieved_ids),
        "relevant_retrieved_at_k": len(relevant_at_k),
        "hit_at_k": 1.0 if relevant_at_k else 0.0,
        # Standard P@k uses k as its denominator. A short result list therefore
        # does not receive the same credit as returning a complete top-k list.
        "precision_at_k": len(relevant_at_k) / k,
        "recall_at_k": len(relevant_at_k) / len(reference_ids),
        "first_relevant_rank": first_relevant_rank,
        "reciprocal_rank": 0.0 if first_relevant_rank is None else 1 / first_relevant_rank,
        "missing_reference_chunk_ids": sorted(reference_ids - set(retrieved_ids[:k])),
    }


def aggregate_rulebook_scores(scores: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return macro averages over already-scored Rulebook cases."""
    records = list(scores)
    if not records:
        raise MetricInputError("At least one Rulebook score is required")
    return {
        "case_count": len(records),
        "hit_at_k": _mean([float(record["hit_at_k"]) for record in records]),
        "precision_at_k": _mean(
            [float(record["precision_at_k"]) for record in records]
        ),
        "recall_at_k": _mean([float(record["recall_at_k"]) for record in records]),
        "mrr": _mean([float(record["reciprocal_rank"]) for record in records]),
    }


def aggregate_router_scores(scores: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute multiclass routing metrics from explicit expected/predicted labels."""
    records = list(scores)
    if not records:
        raise MetricInputError("At least one Router score is required")
    pairs: list[tuple[str, str]] = []
    for index, record in enumerate(records, 1):
        expected = record.get("expected_route")
        predicted = record.get("predicted_route")
        if not isinstance(expected, str) or not expected.strip():
            raise MetricInputError(f"Router score {index} has no expected_route")
        if not isinstance(predicted, str) or not predicted.strip():
            raise MetricInputError(f"Router score {index} has no predicted_route")
        pairs.append((expected.strip().casefold(), predicted.strip().casefold()))

    labels = sorted({label for pair in pairs for label in pair})
    confusion = {
        expected: {
            predicted: sum(1 for actual in pairs if actual == (expected, predicted))
            for predicted in labels
        }
        for expected in labels
    }
    per_route: dict[str, dict[str, float | int]] = {}
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(
            confusion[other][label] for other in labels if other != label
        )
        false_negative = sum(
            confusion[label][other] for other in labels if other != label
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_route[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[label].values()),
        }

    return {
        "case_count": len(pairs),
        "accuracy": sum(expected == predicted for expected, predicted in pairs) / len(pairs),
        "macro_f1": _mean([float(metrics["f1"]) for metrics in per_route.values()]),
        "labels": labels,
        "per_route": per_route,
        "confusion_matrix": confusion,
    }


_NUMERIC_TEXT = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_MOJIBAKE_MARKERS = ("Ã", "Â", "â", "ð", "\ufffd", "\x80", "\x99")


def _normalise_text(value: str) -> str:
    text = unicodedata.normalize("NFC", value).strip()
    if any(marker in text for marker in _MOJIBAKE_MARKERS):
        try:
            repaired = text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        else:
            if sum(text.count(marker) for marker in _MOJIBAKE_MARKERS) > sum(
                repaired.count(marker) for marker in _MOJIBAKE_MARKERS
            ):
                text = unicodedata.normalize("NFC", repaired)
    return text


def _normalise_number(value: int | float | Decimal | str) -> str:
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise MetricInputError(f"Invalid numeric result value: {value!r}") from error
    if not number.is_finite():
        raise MetricInputError("NaN and infinite SQL values cannot be compared")
    if number == 0:
        return "0"
    return format(number.normalize(), "f")


def _normalise_cell(value: Any) -> tuple[str, Any]:
    """Return a hashable, type-aware representation of one SQL cell."""
    if value is None:
        return ("null", None)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float, Decimal)):
        if isinstance(value, float) and not math.isfinite(value):
            raise MetricInputError("NaN and infinite SQL values cannot be compared")
        return ("number", _normalise_number(value))
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, Mapping) and set(value) == {"$bytes_hex"}:
        hexadecimal = value["$bytes_hex"]
        if not isinstance(hexadecimal, str):
            raise MetricInputError("$bytes_hex must be a string")
        return ("bytes", hexadecimal.casefold())
    if isinstance(value, str):
        text = _normalise_text(value)
        if _NUMERIC_TEXT.fullmatch(text):
            return ("number", _normalise_number(text))
        return ("text", text)
    raise MetricInputError(f"Unsupported SQL result value: {type(value).__name__}")


def _normalise_columns(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise MetricInputError(f"{field} must be a list of column names")
    return tuple(_normalise_text(item).casefold() for item in value)


def _normalise_rows(value: Any, width: int, field: str) -> list[tuple[tuple[str, Any], ...]]:
    if not isinstance(value, (list, tuple)):
        raise MetricInputError(f"{field} must be a list of rows")
    rows: list[tuple[tuple[str, Any], ...]] = []
    for index, row in enumerate(value, 1):
        if not isinstance(row, (list, tuple)):
            raise MetricInputError(f"{field} row {index} must be a list or tuple")
        if len(row) != width:
            raise MetricInputError(f"{field} row {index} does not match the column count")
        rows.append(tuple(_normalise_cell(cell) for cell in row))
    return rows


def score_sql_case(
    case: Mapping[str, Any],
    actual_result: Mapping[str, Any],
    *,
    latency_ms: float | None = None,
) -> dict[str, Any]:
    """Score one Text-to-SQL result by status and returned database values."""
    case_id = case.get("case_id")
    expected_status = case.get("expected_status")
    actual_status = actual_result.get("status")
    if not isinstance(case_id, str) or not isinstance(expected_status, str):
        raise MetricInputError("SQL case requires case_id and expected_status")
    if not isinstance(actual_status, str):
        raise MetricInputError(f"{case_id}: actual SQL status must be a string")
    if latency_ms is not None and (
        isinstance(latency_ms, bool)
        or not isinstance(latency_ms, (int, float))
        or latency_ms < 0
    ):
        raise MetricInputError(f"{case_id}: latency_ms must be non-negative or null")

    status_correct = actual_status == expected_status
    columns_correct: bool | None = None
    column_count_correct: bool | None = None
    rows_correct: bool | None = None
    execution_correct: bool | None = None

    if expected_status == "ok":
        expected_columns = _normalise_columns(case.get("expected_columns"), "expected_columns")
        actual_columns = _normalise_columns(actual_result.get("columns", []), "actual columns")
        expected_rows = _normalise_rows(
            case.get("expected_rows"), len(expected_columns), "expected_rows"
        )
        actual_rows = _normalise_rows(
            actual_result.get("rows", []), len(actual_columns), "actual rows"
        )
        column_count_correct = len(actual_columns) == len(expected_columns)
        # Generated SQL may use a different but harmless output alias (for
        # example, ``model_name`` instead of ``unit_name``). Keep exact column
        # equality as diagnostic information, but score execution equivalence
        # from the result shape and values rather than alias spelling.
        columns_correct = actual_columns == expected_columns
        if column_count_correct:
            if bool(case.get("order_sensitive")):
                rows_correct = actual_rows == expected_rows
            else:
                # Counter preserves duplicate-row multiplicity while ignoring order.
                rows_correct = Counter(actual_rows) == Counter(expected_rows)
        else:
            rows_correct = False
        execution_correct = status_correct and column_count_correct and rows_correct
        result_correct = execution_correct
    elif expected_status == "empty":
        actual_rows = actual_result.get("rows", [])
        if not isinstance(actual_rows, (list, tuple)):
            raise MetricInputError(f"{case_id}: actual rows must be a list")
        rows_correct = len(actual_rows) == 0
        result_correct = status_correct and rows_correct
    else:
        result_correct = status_correct

    return {
        "case_id": case_id,
        "expected_status": expected_status,
        "actual_status": actual_status,
        "status_correct": status_correct,
        "columns_correct": columns_correct,
        "column_count_correct": column_count_correct,
        "rows_correct": rows_correct,
        "execution_correct": execution_correct,
        "result_correct": result_correct,
        "is_error": actual_status == "error",
        "latency_ms": latency_ms,
    }


def aggregate_sql_scores(scores: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate status, execution, error, and available latency measurements."""
    records = list(scores)
    if not records:
        raise MetricInputError("At least one SQL score is required")
    executable = [record for record in records if record.get("execution_correct") is not None]
    latencies = [float(record["latency_ms"]) for record in records if record.get("latency_ms") is not None]
    return {
        "case_count": len(records),
        "execution_case_count": len(executable),
        "execution_accuracy": (
            _mean([1.0 if record["execution_correct"] else 0.0 for record in executable])
            if executable
            else None
        ),
        "status_accuracy": _mean(
            [1.0 if record["status_correct"] else 0.0 for record in records]
        ),
        "result_accuracy": _mean(
            [1.0 if record["result_correct"] else 0.0 for record in records]
        ),
        "error_rate": _mean([1.0 if record["is_error"] else 0.0 for record in records]),
        "mean_latency_ms": _mean(latencies) if latencies else None,
    }
