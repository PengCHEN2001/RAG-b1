"""Load and validate the frozen evaluation benchmark datasets.

This module performs structural validation only. It does not call a retriever,
an LLM, RAGAS, or the SQLite database. The future evaluation runner can import
``load_benchmarks`` and fail before a run when benchmark data is malformed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATASET_DIR = Path(__file__).resolve().parent / "datasets"
RULEBOOK_PATH = DATASET_DIR / "rulebook_v1.jsonl"
TEXT2SQL_PATH = DATASET_DIR / "text2sql_v1.jsonl"
COMBINED_PATH = DATASET_DIR / "combined_v1.jsonl"

RULEBOOK_TYPES = {
    "single_hop_specific",
    "multi_hop_specific",
    "multi_hop_abstract",
}
SQL_STATUSES = {"ok", "empty", "needs_clarification", "unsupported"}


class DatasetValidationError(ValueError):
    """Raised when a benchmark file is invalid or internally inconsistent."""


@dataclass(frozen=True)
class BenchmarkDatasets:
    rulebook: tuple[dict[str, Any], ...]
    text2sql: tuple[dict[str, Any], ...]
    combined: tuple[dict[str, Any], ...]

    @property
    def total_cases(self) -> int:
        return len(self.rulebook) + len(self.text2sql) + len(self.combined)


def _error(path: Path, case_id: str, message: str) -> DatasetValidationError:
    return DatasetValidationError(f"{path.name}:{case_id}: {message}")


def _require_fields(
    case: dict[str, Any], required: set[str], path: Path, case_id: str
) -> None:
    missing = sorted(required - case.keys())
    if missing:
        raise _error(path, case_id, f"missing required fields: {', '.join(missing)}")


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_list(value: Any, *, allow_empty: bool = False) -> bool:
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(_non_empty_string(item) for item in value)
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSON object per non-empty line and report precise failures."""
    if not path.is_file():
        raise DatasetValidationError(f"Dataset file not found: {path}")

    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as error:
            raise DatasetValidationError(
                f"{path.name}:{line_number}: invalid JSON: {error.msg}"
            ) from error
        if not isinstance(case, dict):
            raise DatasetValidationError(
                f"{path.name}:{line_number}: each line must contain a JSON object"
            )
        cases.append(case)

    if not cases:
        raise DatasetValidationError(f"Dataset file is empty: {path}")
    return cases


def _validate_ids(
    cases: list[dict[str, Any]], path: Path, pattern: str
) -> set[str]:
    identifiers: set[str] = set()
    for index, case in enumerate(cases, 1):
        case_id = case.get("case_id")
        if not _non_empty_string(case_id):
            raise DatasetValidationError(
                f"{path.name}:{index}: case_id must be a non-empty string"
            )
        if re.fullmatch(pattern, case_id) is None:
            raise _error(path, case_id, f"case_id must match {pattern}")
        if case_id in identifiers:
            raise _error(path, case_id, "duplicate case_id")
        identifiers.add(case_id)
    return identifiers


def validate_rulebook(
    cases: list[dict[str, Any]], path: Path = RULEBOOK_PATH
) -> set[str]:
    identifiers = _validate_ids(cases, path, r"rb_\d{3}")
    required = {
        "case_id",
        "question",
        "question_type",
        "reference_answer",
        "reference_contexts",
        "review_status",
    }
    for case in cases:
        case_id = case["case_id"]
        _require_fields(case, required, path, case_id)
        if not _non_empty_string(case["question"]):
            raise _error(path, case_id, "question must be a non-empty string")
        if case["question_type"] not in RULEBOOK_TYPES:
            raise _error(path, case_id, "unsupported question_type")
        if not _non_empty_string(case["reference_answer"]):
            raise _error(path, case_id, "reference_answer must be a non-empty string")
        if case["review_status"] != "approved":
            raise _error(path, case_id, "review_status must be approved")

        contexts = case["reference_contexts"]
        if not isinstance(contexts, list) or not contexts:
            raise _error(path, case_id, "reference_contexts must be a non-empty list")
        chunk_ids: set[str] = set()
        for context in contexts:
            if not isinstance(context, dict):
                raise _error(path, case_id, "each reference context must be an object")
            _require_fields(context, {"chunk_id", "text", "source", "pages"}, path, case_id)
            chunk_id = context["chunk_id"]
            if not _non_empty_string(chunk_id) or not _non_empty_string(context["text"]):
                raise _error(path, case_id, "reference context ID and text must be non-empty")
            if chunk_id in chunk_ids:
                raise _error(path, case_id, f"duplicate reference chunk: {chunk_id}")
            chunk_ids.add(chunk_id)
            if not isinstance(context["pages"], list) or not all(
                isinstance(page, int) and page > 0 for page in context["pages"]
            ):
                raise _error(path, case_id, "reference context pages must be positive integers")
    return identifiers


def validate_text2sql(
    cases: list[dict[str, Any]], path: Path = TEXT2SQL_PATH
) -> set[str]:
    identifiers = _validate_ids(cases, path, r"sql_\d{3}")
    required = {
        "case_id",
        "question",
        "category",
        "expected_status",
        "gold_sql",
        "expected_columns",
        "expected_rows",
        "order_sensitive",
        "review_status",
    }
    for case in cases:
        case_id = case["case_id"]
        _require_fields(case, required, path, case_id)
        if not _non_empty_string(case["question"]):
            raise _error(path, case_id, "question must be a non-empty string")
        if not _non_empty_string(case["category"]):
            raise _error(path, case_id, "category must be a non-empty string")
        status = case["expected_status"]
        if status not in SQL_STATUSES:
            raise _error(path, case_id, f"unsupported expected_status: {status!r}")
        if case["review_status"] != "approved":
            raise _error(path, case_id, "review_status must be approved")
        if not isinstance(case["order_sensitive"], bool):
            raise _error(path, case_id, "order_sensitive must be a boolean")
        if not _string_list(case["expected_columns"], allow_empty=True):
            raise _error(path, case_id, "expected_columns must be a list of strings")
        if not isinstance(case["expected_rows"], list) or not all(
            isinstance(row, list) for row in case["expected_rows"]
        ):
            raise _error(path, case_id, "expected_rows must be a list of row lists")
        if any(len(row) != len(case["expected_columns"]) for row in case["expected_rows"]):
            raise _error(path, case_id, "expected row width does not match expected_columns")

        if status in {"ok", "empty"}:
            if not _non_empty_string(case["gold_sql"]):
                raise _error(path, case_id, f"{status} cases require gold_sql")
            if not case["expected_columns"]:
                raise _error(path, case_id, f"{status} cases require expected_columns")
            if status == "ok" and not case["expected_rows"]:
                raise _error(path, case_id, "ok cases require at least one expected row")
            if status == "empty" and case["expected_rows"]:
                raise _error(path, case_id, "empty cases cannot contain expected rows")
        elif case["gold_sql"] is not None:
            raise _error(path, case_id, f"{status} cases must not define gold_sql")

        if status == "needs_clarification" and not _string_list(
            case.get("expected_candidates")
        ):
            raise _error(path, case_id, "needs_clarification requires expected_candidates")
    return identifiers


def validate_combined(
    cases: list[dict[str, Any]],
    *,
    sql_ids: set[str],
    rulebook_ids: set[str],
    path: Path = COMBINED_PATH,
) -> set[str]:
    identifiers = _validate_ids(cases, path, r"combined_\d{3}")
    required = {
        "case_id",
        "question",
        "category",
        "expected_route",
        "required_sources",
        "sql_reference_case_ids",
        "rulebook_reference_case_ids",
        "reference_answer",
        "review_status",
    }
    for case in cases:
        case_id = case["case_id"]
        _require_fields(case, required, path, case_id)
        if not _non_empty_string(case["question"]):
            raise _error(path, case_id, "question must be a non-empty string")
        if case["expected_route"] != "both":
            raise _error(path, case_id, "expected_route must be both")
        if not _string_list(case["required_sources"]):
            raise _error(path, case_id, "required_sources must be a non-empty string list")
        if set(case["required_sources"]) != {"sql", "rulebook"}:
            raise _error(path, case_id, "required_sources must contain sql and rulebook")
        if not _string_list(case["sql_reference_case_ids"]):
            raise _error(path, case_id, "sql_reference_case_ids must be non-empty")
        if not _string_list(case["rulebook_reference_case_ids"]):
            raise _error(path, case_id, "rulebook_reference_case_ids must be non-empty")

        missing_sql = sorted(set(case["sql_reference_case_ids"]) - sql_ids)
        missing_rulebook = sorted(set(case["rulebook_reference_case_ids"]) - rulebook_ids)
        if missing_sql:
            raise _error(path, case_id, f"unknown SQL references: {', '.join(missing_sql)}")
        if missing_rulebook:
            raise _error(
                path, case_id, f"unknown Rulebook references: {', '.join(missing_rulebook)}"
            )
        if not _non_empty_string(case["reference_answer"]):
            raise _error(path, case_id, "reference_answer must be a non-empty string")
        if case["review_status"] != "approved":
            raise _error(path, case_id, "review_status must be approved")
    return identifiers


def load_benchmarks(dataset_dir: Path = DATASET_DIR) -> BenchmarkDatasets:
    """Load all v1 datasets and validate their cross-dataset references."""
    rulebook_path = dataset_dir / RULEBOOK_PATH.name
    text2sql_path = dataset_dir / TEXT2SQL_PATH.name
    combined_path = dataset_dir / COMBINED_PATH.name

    rulebook = load_jsonl(rulebook_path)
    text2sql = load_jsonl(text2sql_path)
    combined = load_jsonl(combined_path)

    rulebook_ids = validate_rulebook(rulebook, rulebook_path)
    sql_ids = validate_text2sql(text2sql, text2sql_path)
    combined_ids = validate_combined(
        combined,
        sql_ids=sql_ids,
        rulebook_ids=rulebook_ids,
        path=combined_path,
    )

    all_ids = rulebook_ids | sql_ids | combined_ids
    expected_count = len(rulebook_ids) + len(sql_ids) + len(combined_ids)
    if len(all_ids) != expected_count:
        raise DatasetValidationError("case_id values must be unique across all datasets")

    return BenchmarkDatasets(tuple(rulebook), tuple(text2sql), tuple(combined))


def main() -> None:
    datasets = load_benchmarks()
    print(
        "Validated "
        f"{len(datasets.rulebook)} Rulebook, "
        f"{len(datasets.text2sql)} Text-to-SQL, and "
        f"{len(datasets.combined)} Combined cases "
        f"({datasets.total_cases} total)."
    )


if __name__ == "__main__":
    main()
