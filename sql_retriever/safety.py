"""SQLite-native access control and bounded read-only execution."""

import json
import math
import re
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from .result import CompilationFailure, Parameter, PolicyViolation, RetrievalFailure

# Access policy is independent of the model's reference document.
ALLOWED_TABLES = {
    "abilities", "datasheets", "datasheets_abilities",
    "datasheets_detachment_abilities", "datasheets_enhancements",
    "datasheets_keywords", "datasheets_leader", "datasheets_models",
    "datasheets_models_cost", "datasheets_options", "datasheets_stratagems",
    "datasheets_unit_composition", "datasheets_wargear", "detachment_abilities",
    "detachments", "detachments_chapter_dp", "enhancements", "factions",
    "last_update", "source", "stratagems",
}
FUNCTIONS = {"count", "sum", "avg", "min", "max", "coalesce", "nullif", "lower",
             "upper", "trim", "length", "abs", "round", "like", "glob"}


@dataclass(frozen=True)
class ExecutionLimits:
    rows: int = 200
    bytes: int = 1024 * 1024
    seconds: float = 2.0
    vm_steps: int = 2_000_000


@dataclass
class ExecutedQuery:
    columns: list[str]
    rows: list[tuple]
    truncated: bool


def validate_parameters(parameters: object) -> dict[str, Parameter]:
    if not isinstance(parameters, dict) or len(parameters) > 100:
        raise PolicyViolation("Parameters must be a bounded named-value object.")
    for name, value in parameters.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
            raise PolicyViolation("Invalid named parameter.")
        if type(value) not in (str, int, float, type(None)):
            raise PolicyViolation("Parameters must be SQLite scalar values.")
        if isinstance(value, float) and not math.isfinite(value):
            raise PolicyViolation("Non-finite parameters are forbidden.")
        if isinstance(value, int) and not -(2**63) <= value < 2**63:
            raise PolicyViolation("Integer parameter exceeds SQLite range.")
        if isinstance(value, str):
            try:
                size = len(value.encode())
            except UnicodeError as exc:
                raise PolicyViolation("Parameter is not valid Unicode text.") from exc
            if size > 16 * 1024:
                raise PolicyViolation("Parameter exceeds the size limit.")
    return parameters


def open_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.1)
    try:
        connection.enable_load_extension(False)
        connection.execute("PRAGMA query_only = ON")
    except Exception:
        connection.close()
        raise
    return connection


def install_authorizer(connection: sqlite3.Connection) -> None:
    def authorize(action, first, second, database, source):
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ and (first or "").lower() in ALLOWED_TABLES:
            # COUNT(*) can report an empty column and no database name.
            if database == "main" or (database is None and not second):
                return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and (second or "").lower() in FUNCTIONS:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    connection.set_authorizer(authorize)


def execute_bounded(path: Path, sql: str, parameters: dict,
                    *, limits: ExecutionLimits = ExecutionLimits()) -> ExecutedQuery:
    """Let SQLite compile one statement, authorize its operations, and bound its results."""
    validate_parameters(parameters)
    if not isinstance(sql, str) or not sql.strip():
        raise PolicyViolation("SQL must be non-empty and at most 16 KiB.")
    try:
        size = len(sql.encode())
    except UnicodeError as exc:
        raise PolicyViolation("SQL is not valid Unicode text.") from exc
    if size > 16 * 1024:
        raise PolicyViolation("SQL must be non-empty and at most 16 KiB.")

    started = time.monotonic()
    steps = 0
    interval = min(1000, max(1, limits.vm_steps))

    def progress():
        nonlocal steps
        steps += interval
        return int(steps >= limits.vm_steps or time.monotonic() - started >= limits.seconds)

    try:
        with closing(open_readonly(path)) as connection:
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, max(1024, limits.bytes))
            connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 16 * 1024)
            install_authorizer(connection)
            connection.set_progress_handler(progress, interval)
            cursor = connection.execute(sql, parameters)
            if cursor.description is None:
                raise PolicyViolation("Query must return a result set.")
            columns = [description[0] for description in cursor.description]
            rows = []
            used = len(json.dumps(columns).encode()) + 2
            if used > limits.bytes:
                raise RetrievalFailure("result_budget", "Column metadata exceeds the result-size budget.")
            truncated = False
            for _ in range(limits.rows + 1):
                row = cursor.fetchone()
                if time.monotonic() - started >= limits.seconds:
                    raise RetrievalFailure("query_budget", "Query execution budget exceeded.")
                if row is None:
                    break
                size = len(json.dumps(row, ensure_ascii=False, default=lambda b: b.hex()).encode())
                size += 2 if rows else 0  # JSON row separators; array brackets are counted above.
                if len(rows) >= limits.rows or used + size > limits.bytes:
                    truncated = True
                    break
                rows.append(row)
                used += size
            return ExecutedQuery(columns, rows, truncated)
    except sqlite3.Error as exc:
        code = getattr(exc, "sqlite_errorcode", 0) & 0xff
        if code in (sqlite3.SQLITE_INTERRUPT, sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
            raise RetrievalFailure("query_budget", "Query execution budget exceeded or database is busy.") from exc
        if code in (sqlite3.SQLITE_AUTH, sqlite3.SQLITE_READONLY) or "not authorized" in str(exc):
            raise PolicyViolation("SQLite denied an unauthorized operation.") from exc
        if isinstance(exc, sqlite3.ProgrammingError):
            raise PolicyViolation("Use one SQL statement with matching named bindings.") from exc
        if code == sqlite3.SQLITE_TOOBIG:
            raise RetrievalFailure("result_budget", "A database value exceeds the result-size budget.") from exc
        if isinstance(exc, sqlite3.OperationalError) and any(term in str(exc) for term in (
                "syntax error", "no such column", "ambiguous column", "misuse of", "wrong number",
                "no such function", "no such table", "HAVING clause")):
            raise CompilationFailure("SQLite compilation failed: " + str(exc)[:300]) from exc
        raise RetrievalFailure("database_error", "SQLite query could not be executed.") from exc
