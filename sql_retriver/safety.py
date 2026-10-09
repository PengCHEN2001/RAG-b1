"""Fail-closed SQL validation and independent bounded SQLite execution."""

import json
import math
import re
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope

from .result import CompilationFailure, Parameter, PolicyViolation, RetrievalFailure
from .schema import SchemaContext, integer_expression, open_readonly

FUNCTIONS = {"count", "sum", "avg", "min", "max", "coalesce", "nullif", "lower",
             "upper", "trim", "length", "abs", "round", "like", "glob"}
ALLOWED_NODES = {
    "Select", "With", "CTE", "Subquery", "From", "Table", "TableAlias", "Identifier",
    "Column", "Alias", "Join", "Where", "Group", "Having", "Order", "Ordered",
    "Limit", "Offset", "Distinct", "Placeholder", "Literal", "Null", "Star", "Paren",
    "And", "Or", "Not", "EQ", "NEQ", "GT", "GTE", "LT", "LTE", "Is", "In",
    "Between", "Exists", "Like", "Glob", "Case", "If", "Cast", "DataType",
    "Count", "Sum", "Avg", "Min", "Max", "Coalesce", "Nullif", "Lower", "Upper",
    "Trim", "Length", "Abs", "Round", "Anonymous", "Neg",
}
NUMERIC_FIELDS = {
    "datasheets_models": {"m", "t", "sv", "invsv", "w", "ld", "oc"},
    "datasheets_models_cost": {"cost"},
    "datasheets_wargear": {"range", "a", "bsws", "s", "ap", "d"},
    "stratagems": {"cpcost"}, "enhancements": {"cost"},
}


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


def _projection(scope: Scope, name: str) -> exp.Expression | None:
    for item in scope.expression.expressions:
        if item.alias_or_name.lower() == name.lower():
            return item.this if isinstance(item, exp.Alias) else item
    return None


def _source(scope: Scope, column: exp.Column):
    current = scope
    while current is not None:
        source = current.sources.get(column.table)
        if source is not None:
            return current, source
        current = current.parent
    return scope, None


def _origin(scope: Scope, column: exp.Column) -> tuple[str, str] | None:
    """Trace only unmodified projected keys through CTEs/derived tables."""
    if not column.table:
        projected = _projection(scope, column.name)
        if isinstance(projected, exp.Column) and projected.table:
            return _origin(scope, projected)
        return None
    _, source = _source(scope, column)
    if isinstance(source, exp.Table):
        return source.name.lower(), column.name.lower()
    if isinstance(source, Scope):
        projected = _projection(source, column.name)
        if isinstance(projected, exp.Column):
            return _origin(source, projected)
    return None


def _numeric(scope: Scope, node: exp.Expression, schema: SchemaContext,
             integer_cases: set[int]) -> bool:
    if isinstance(node, (exp.Alias, exp.Paren, exp.Neg)):
        return _numeric(scope, node.this, schema, integer_cases)
    if id(node) in integer_cases or isinstance(node, (exp.Count, exp.Length)):
        return True
    if isinstance(node, exp.Literal):
        return not node.is_string
    if isinstance(node, exp.Column):
        if not node.table:
            item = _projection(scope, node.name)
            return item is not None and item != node and _numeric(scope, item, schema, integer_cases)
        _, source = _source(scope, node)
        if isinstance(source, Scope):
            item = _projection(source, node.name)
            return item is not None and _numeric(source, item, schema, integer_cases)
        origin = _origin(scope, node)
        if origin:
            table, column = origin
            types = {key.lower(): value.upper() for key, value in schema.tables[table].items()}
            return any(t in types.get(column, "") for t in ("INT", "REAL", "NUM", "FLOAT", "DOUBLE"))
    if isinstance(node, (exp.Sum, exp.Avg, exp.Min, exp.Max, exp.Abs, exp.Round)):
        return _numeric(scope, node.this, schema, integer_cases)
    if isinstance(node, (exp.Coalesce, exp.Nullif)):
        return _numeric(scope, node.this, schema, integer_cases)
    return False


def _raw_statistic(scope: Scope, node: exp.Expression, integer_cases: set[int]) -> bool:
    for column in node.find_all(exp.Column):
        if _inside(column, integer_cases):
            continue
        origin = _origin(scope, column)
        if origin and origin[1] in NUMERIC_FIELDS.get(origin[0], set()):
            return True
    return False


def _inside(node: exp.Expression, ids: set[int]) -> bool:
    while node is not None:
        if id(node) in ids:
            return True
        node = node.parent
    return False


def validate_sql(sql: str, parameters: dict, schema: SchemaContext) -> str:
    """Return canonical, qualified SQL only after every policy check passes."""
    validate_parameters(parameters)
    if not isinstance(sql, str) or not sql.strip():
        raise PolicyViolation("SQL must be non-empty and at most 16 KiB.")
    try:
        size = len(sql.encode())
    except UnicodeError as exc:
        raise PolicyViolation("SQL is not valid Unicode text.") from exc
    if size > 16 * 1024:
        raise PolicyViolation("SQL must be non-empty and at most 16 KiB.")
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except (ParseError, RecursionError) as exc:
        raise CompilationFailure("SQL parsing failed.") from exc
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise PolicyViolation("Exactly one SELECT statement is required.")
    tree = statements[0]
    for node in tree.walk():
        if type(node).__name__ not in ALLOWED_NODES:
            raise PolicyViolation(f"Unsupported SQL construct: {type(node).__name__}.")
        if isinstance(node, exp.With) and node.args.get("recursive"):
            raise PolicyViolation("Recursive CTEs are forbidden.")
        if isinstance(node, exp.Table) and (node.db or node.catalog or not isinstance(node.this, exp.Identifier)):
            raise PolicyViolation("Only application tables in the main database are allowed.")
        if isinstance(node, exp.Star) and not isinstance(node.parent, exp.Count):
            raise PolicyViolation("Project explicit columns, not SELECT *.")
        if isinstance(node, exp.Select):
            depth = 1
            parent = node.parent
            while parent is not None:
                depth += isinstance(parent, exp.Select)
                parent = parent.parent
            if depth > 3:
                raise PolicyViolation("SELECT nesting exceeds three levels.")
            if node.args.get("into") or node.args.get("locks"):
                raise PolicyViolation("SELECT side effects are forbidden.")
        if isinstance(node, exp.Func) and not isinstance(node, (
                exp.Case, exp.If, exp.Cast, exp.And, exp.Or, exp.Exists)):
            name = node.name.lower() if isinstance(node, exp.Anonymous) else node.sql_name().lower()
            if name not in FUNCTIONS:
                raise PolicyViolation("SQL function is not approved.")
        if isinstance(node, exp.If) and not isinstance(node.parent, exp.Case):
            raise PolicyViolation("Only CASE branches are supported.")
    bindings = list(tree.find_all(exp.Placeholder))
    if any(not p.name or p.args.get("kind") for p in bindings):
        raise PolicyViolation("Use only named :parameters.")
    if {p.name for p in bindings} != set(parameters):
        raise PolicyViolation("SQL bindings and parameter names must match exactly.")
    if len(list(tree.find_all(exp.Join))) > 4:
        raise PolicyViolation("At most four joins are allowed.")
    cte_names = {cte.alias.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if table.name.lower() not in schema.tables and table.name.lower() not in cte_names:
            raise PolicyViolation("Query references a table outside the approved live schema.")
    try:
        tree = qualify(tree, dialect="sqlite", schema=schema.tables, expand_stars=False,
                       infer_schema=False)
    except (OptimizeError, sqlglot.errors.SqlglotError) as exc:
        raise CompilationFailure("Column or alias resolution failed.") from exc
    except RecursionError as exc:
        raise PolicyViolation("Expression nesting exceeds validator capacity.") from exc

    integer_cases = set()
    for case in tree.find_all(exp.Case):
        columns = list(case.find_all(exp.Column))
        if columns:
            template = sqlglot.parse_one(integer_expression(columns[0].sql("sqlite")), read="sqlite")
            if case == template:
                integer_cases.add(id(case))
    for cast in tree.find_all(exp.Cast):
        if not _inside(cast, integer_cases):
            raise PolicyViolation("CAST requires the reviewed guarded integer CASE template.")
    for literal in tree.find_all(exp.Literal):
        if _inside(literal, integer_cases):
            continue
        parent = literal.parent
        if isinstance(parent, (exp.Limit, exp.Offset)):
            if literal.is_string or not literal.this.isdigit() or int(literal.this) > 1_000_000:
                raise PolicyViolation("LIMIT/OFFSET must be bounded non-negative integers.")
        elif (literal.is_string and literal.this == "" and isinstance(parent, exp.Nullif)
              and parent.expression is literal):
            continue
        else:
            raise PolicyViolation("Bind user values; only reviewed structural literals are allowed.")
    for limit in (*tree.find_all(exp.Limit), *tree.find_all(exp.Offset)):
        if not isinstance(limit.expression, exp.Literal):
            raise PolicyViolation("LIMIT/OFFSET must be structural integer literals.")

    scopes = list(traverse_scope(tree))
    if not any(isinstance(source, exp.Table) for s in scopes for source in s.sources.values()):
        raise PolicyViolation("Query must read application data.")
    for scope in scopes:
        source_clause = scope.expression.args.get("from_")
        joined = {source_clause.this.alias_or_name} if source_clause is not None else set()
        for join in scope.expression.args.get("joins", []):
            if join.side.upper() not in ("", "LEFT") or join.kind.upper() not in ("", "INNER"):
                raise PolicyViolation("Only INNER and LEFT joins are supported.")
            on = join.args.get("on")
            if on is None or join.args.get("using") or any(on.find_all(exp.Or)):
                raise PolicyViolation("Joins require conjunctive explicit ON key equalities.")
            target = join.this.alias_or_name
            connected = False
            for equality in on.find_all(exp.EQ):
                left, right = equality.this, equality.expression
                if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                    continue
                if left.table == right.table:
                    continue
                origins = (_origin(scope, left), _origin(scope, right))
                if None in origins or frozenset(origins) not in schema.edges:
                    raise PolicyViolation("Join equality is not a reviewed relationship.")
                if ((left.table == target and right.table in joined)
                        or (right.table == target and left.table in joined)):
                    connected = True
            if not connected:
                raise PolicyViolation("Each join must connect through a reviewed key equality.")
            joined.add(target)
        # Scope.walk excludes nested scopes, avoiding incorrect correlated-column ownership.
        for node in scope.walk():
            if _inside(node, integer_cases):
                continue
            if isinstance(node, (exp.Sum, exp.Avg, exp.Abs, exp.Round)):
                if not _numeric(scope, node.this, schema, integer_cases):
                    raise PolicyViolation("Numeric operations require guarded numeric inputs.")
            if isinstance(node, exp.EQ) and isinstance(node.this, exp.Column) and isinstance(
                    node.expression, exp.Column) and node.this.table != node.expression.table:
                origins = (_origin(scope, node.this), _origin(scope, node.expression))
                if None in origins or frozenset(origins) not in schema.edges:
                    raise PolicyViolation("Correlated equalities require reviewed relationship keys.")
            if isinstance(node, (exp.LT, exp.LTE, exp.GT, exp.GTE, exp.Between)):
                operands = ((node.this, node.args["low"], node.args["high"])
                            if isinstance(node, exp.Between) else (node.this, node.expression))
                for operand in operands:
                    if (isinstance(operand, exp.Column) or _raw_statistic(scope, operand, integer_cases)) and not _numeric(
                            scope, operand, schema, integer_cases):
                        raise PolicyViolation("Numeric comparisons of TEXT require guarded conversion.")
            if isinstance(node, (exp.EQ, exp.NEQ)):
                left, right = node.this, node.expression
                for column, value in ((left, right), (right, left)):
                    if (isinstance(column, exp.Column) and isinstance(value, exp.Placeholder)
                            and type(parameters[value.name]) in (int, float)
                            and not _numeric(scope, column, schema, integer_cases)):
                        raise PolicyViolation("Numeric equality against TEXT requires guarded conversion.")
            if isinstance(node, (exp.Ordered, exp.Min, exp.Max)):
                if _raw_statistic(scope, node.this, integer_cases) and not _numeric(
                        scope, node.this, schema, integer_cases):
                    raise PolicyViolation("Numeric statistics require guarded sorting/aggregation.")
    return tree.sql(dialect="sqlite")


def install_authorizer(connection: sqlite3.Connection, schema: SchemaContext) -> None:
    approved = {table.lower(): {column.lower() for column in columns}
                for table, columns in schema.tables.items()}

    def authorize(action, first, second, database, source):
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            columns = approved.get((first or "").lower())
            # SQLite's COUNT(*) optimization reports an empty column and no database.
            if (database == "main" or (database is None and not second)) and columns is not None and (
                    not second or second.lower() in columns):
                return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and (second or "").lower() in FUNCTIONS:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    connection.set_authorizer(authorize)


def execute_bounded(path: Path, sql: str, parameters: dict, schema: SchemaContext,
                    *, limits: ExecutionLimits = ExecutionLimits(), compile_only=False) -> ExecutedQuery:
    """Execute exactly one statement, even when the AST validator is bypassed."""
    validate_parameters(parameters)
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
            connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 32 * 1024)
            install_authorizer(connection, schema)
            connection.set_progress_handler(progress, interval)
            # Preparing EXPLAIN compiles without running the SELECT or exposing rows.
            connection.execute("EXPLAIN " + sql, parameters).close()
            if compile_only:
                return ExecutedQuery([], [], False)
            cursor = connection.execute(sql, parameters)
            columns = [description[0] for description in cursor.description]
            rows = []
            used = len(json.dumps(columns).encode()) + 2
            truncated = False
            for _ in range(limits.rows + 1):
                row = cursor.fetchone()
                if row is None:
                    break
                if time.monotonic() - started >= limits.seconds:
                    raise RetrievalFailure("query_budget", "Query execution budget exceeded.")
                size = len(json.dumps(row, ensure_ascii=False, default=lambda b: b.hex()).encode()) + 1
                if len(rows) >= limits.rows or used + size > limits.bytes:
                    truncated = True
                    break
                rows.append(row)
                used += size
            return ExecutedQuery(columns, rows, truncated)
    except sqlite3.Error as exc:
        code = getattr(exc, "sqlite_errorcode", None)
        if code in (sqlite3.SQLITE_INTERRUPT, sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
            raise RetrievalFailure("query_budget", "Query execution budget exceeded or database is busy.") from exc
        if code in (sqlite3.SQLITE_AUTH, sqlite3.SQLITE_READONLY) or "not authorized" in str(exc):
            raise PolicyViolation("SQLite denied an unauthorized operation.") from exc
        if code == sqlite3.SQLITE_TOOBIG:
            raise RetrievalFailure("result_budget", "A database value exceeds the result-size budget.") from exc
        if isinstance(exc, sqlite3.OperationalError) and any(term in str(exc) for term in (
                "syntax error", "no such column", "ambiguous column", "misuse of", "wrong number",
                "no such function", "no such table", "HAVING clause")):
            raise CompilationFailure() from exc
        raise RetrievalFailure("database_error", "SQLite query could not be executed.") from exc
