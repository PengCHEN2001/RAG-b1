"""Public retrieval outcomes and internal failure categories."""

from dataclasses import dataclass, field
from typing import Literal

SQLiteValue = str | int | float | bytes | None
Parameter = str | int | float | None


@dataclass
class RetrievalResult:
    status: Literal["ok", "empty", "needs_clarification", "unsupported", "error"]
    strategy: Literal["fixed", "generated", "none"] = "none"
    columns: list[str] = field(default_factory=list)
    rows: list[tuple[SQLiteValue, ...]] = field(default_factory=list)
    sql: str | None = None
    parameters: dict[str, Parameter] = field(default_factory=dict)
    truncated: bool = False
    message: str = ""
    error_code: str | None = None
    candidates: list[dict[str, str]] = field(default_factory=list)


class RetrievalFailure(Exception):
    """A failure safe to expose without database values or user text."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class PolicyViolation(RetrievalFailure):
    def __init__(self, message: str):
        super().__init__("sql_policy", message)


class CompilationFailure(RetrievalFailure):
    def __init__(self, message: str = "Query syntax or column resolution failed."):
        super().__init__("sql_compilation", message)
