"""Check resolved identities in filtering predicates, independently of SQL access control."""

from functools import cache

from sqlglot import exp, parse
from sqlglot.errors import OptimizeError, ParseError
from sqlglot.optimizer.scope import Scope, build_scope

from .entities import ENTITY_TABLES, identity_fields, resolved_bindings
from .result import CompilationFailure, PolicyViolation
from .safety import ALLOWED_TABLES, validate_sql

REFERENCE_COLUMNS = {
    "unit": {"datasheetid", "leaderid", "attachedid"},
    "faction": {"factionid"}, "source": {"sourceid"},
    "detachment": {"detachmentid"}, "enhancement": {"enhancementid"},
    "stratagem": {"stratagemid"},
}


def _identity_column(entity: dict[str, str], field: str, table: str, column: str) -> bool:
    if "id" not in entity:
        return table == "datasheets_abilities" and column == {"unit_id": "datasheetid", "line": "line"}[field]
    kind = entity["kind"]
    base, primary, _ = ENTITY_TABLES[kind]
    if kind == "ability":
        return table == base and column == {"id": "id", "faction_id": "factionid"}[field]
    return ((table == base and column == primary.lower())
            or (table in ALLOWED_TABLES and column in REFERENCE_COLUMNS.get(kind, set())))


def validate_entity_constraints(sql: str, parameters: dict, resolved: dict[str, dict[str, str]]) -> None:
    """Accept direct key filters in WHERE/inner JOIN, or in a required input CTE.

    This proves the presence of effective identity filters, not arbitrary query semantics.
    Unproven SQL forms are rejected rather than accepted from textual parameter mentions.
    """
    validate_sql(sql)
    bindings = resolved_bindings(resolved)
    if any(parameters.get(name) != value for name, value in bindings.items()):
        raise PolicyViolation("Generated bindings must preserve all resolved entity identities.")
    if not resolved:
        return
    groups = {f"{key}_{field}": (entity["kind"], field)
              for key, entity in resolved.items() for field in identity_fields(entity)}

    def atoms(node):
        if isinstance(node, exp.Paren):
            return atoms(node.this)
        if isinstance(node, exp.And):
            return atoms(node.this) | atoms(node.expression)
        if isinstance(node, exp.Or):
            return atoms(node.this) & atoms(node.expression)
        pairs = []
        if isinstance(node, exp.EQ):
            pairs = [(node.this, node.expression), (node.expression, node.this)]
        elif isinstance(node, exp.In) and not node.args.get("query") and node.expressions:
            names = [value.name for value in node.expressions if isinstance(value, exp.Placeholder)]
            group = groups.get(names[0]) if names else None
            # A list of resolved single-key identities is safe; compound card keys need paired equalities.
            if (len(names) == len(node.expressions) and group is not None
                    and group[1] == "id" and group[0] != "ability"
                    and all(groups.get(name) == group for name in names)):
                pairs = [(node.this, value) for value in node.expressions]
        return {(column.table.lower(), column.name.lower(), value.name)
                for column, value in pairs
                if isinstance(column, exp.Column) and isinstance(value, exp.Placeholder)}

    @cache
    def guaranteed(scope):
        if scope.set_operation_scopes:
            return set.intersection(*(guaranteed(branch) for branch in scope.set_operation_scopes))
        select = scope.expression
        if not isinstance(select, exp.Select):
            raise PolicyViolation("Entity constraints require a SELECT query.")
        joins = select.args.get("joins", [])
        if any(join.side in ("RIGHT", "FULL") for join in joins):
            raise PolicyViolation("Use WHERE identity filters with FROM, inner joins or left joins.")
        sources = {alias.lower(): source for alias, (_, source) in scope.selected_sources.items()}
        optional = {join.alias_or_name.lower() for join in joins if join.side == "LEFT"}
        found = set()
        for alias, source in sources.items():
            if isinstance(source, Scope) and alias not in optional:
                found.update(guaranteed(source))
        filters = atoms(select.args.get("where").this) if select.args.get("where") else set()
        for join in joins:
            if not join.side and join.args.get("on"):
                filters.update(atoms(join.args["on"]))
        by_alias = {}
        for alias, column, name in filters:
            if not alias and len(sources) == 1:
                alias = next(iter(sources))
            source = sources.get(alias)
            if isinstance(source, exp.Table):
                by_alias.setdefault(alias, []).append((source.name.lower(), column, name))
        for key, entity in resolved.items():
            fields = identity_fields(entity)
            for columns in by_alias.values():
                matched = {field for field in fields
                           if any(name == f"{key}_{field}" and _identity_column(entity, field, table, column)
                                  for table, column, name in columns)}
                if len(matched) == len(fields):
                    found.add(key)
                    break
        return found

    try:
        statements = [statement for statement in parse(sql, read="sqlite") if statement is not None]
        if len(statements) != 1:
            raise PolicyViolation("Use one SQL statement with resolved identity filters.")
        scope = build_scope(statements[0])
        if scope is None:
            raise PolicyViolation("Entity constraints require a SELECT query.")
        missing = resolved.keys() - guaranteed(scope)
        if missing:
            raise PolicyViolation("Missing effective identity filters for: " + ", ".join(sorted(missing)))
    except (ParseError, OptimizeError) as exc:
        raise CompilationFailure("SQL parsing or name resolution failed: " + str(exc)[:300]) from exc
    except RecursionError as exc:
        raise PolicyViolation("SQL nesting exceeds identity validation limits.") from exc
