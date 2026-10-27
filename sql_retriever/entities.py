"""Resolve explicitly mentioned entities without model-chosen database IDs."""

import re
import sys
import unicodedata
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Literal

from .result import RetrievalFailure
from .safety import ExecutedQuery, ExecutionLimits, execute_bounded

ENTITY_TABLES = {
    "unit": ("datasheets", "Id", {"faction": "factionId", "source": "sourceId"}),
    "faction": ("factions", "Id", {}),
    "detachment": ("detachments", "Id", {"faction": "factionId"}),
    "ability": ("abilities", "Id", {"faction": "factionId", "source": "sourceId", "unit": "unitId"}),
    "enhancement": ("enhancements", "id", {"faction": "FactionId"}),
    "stratagem": ("stratagems", "id", {"faction": "FactionId"}),
    "source": ("source", "Id", {}),
}

# Catalogs retain byte/time budgets, without the business-query row cap.
NAME_LOOKUP_LIMITS = ExecutionLimits(rows=sys.maxsize)
FUZZY_CANDIDATE_LIMIT = 5

ABILITY_ENTITIES = """
    SELECT a.Id, a.name, a.factionId, NULL AS sourceId,
           NULL AS unitId, NULL AS line, NULL AS ability_id
    FROM abilities a
    UNION ALL
    SELECT NULL, COALESCE(NULLIF(da.name, ''), a.name), d.factionId, d.sourceId,
           da.DatasheetId, da.line, NULLIF(da.abilityId, '')
    FROM datasheets_abilities da
    JOIN datasheets d ON d.Id = da.DatasheetId
    LEFT JOIN abilities a ON a.Id = da.abilityId
        AND (a.factionId = d.factionId OR COALESCE(a.factionId, '') = '')
    {card_filter}
"""


@dataclass
class EntityResolution:
    """Verified identities and any clarification needed by the next lookup."""

    status: Literal["resolved", "needs_clarification"] = "resolved"
    resolved: dict[str, dict[str, str]] = field(default_factory=dict)
    candidates: list[dict[str, str]] = field(default_factory=list)
    message: str = ""
    truncated: bool = False


def validate_entities(entities: object, question: str) -> list[dict[str, str]]:
    if not isinstance(entities, list) or len(entities) > 8:
        raise RetrievalFailure("invalid_model_output", "Entities must be a bounded list.")
    keys = set()
    required = {"key", "kind", "match", "value"}
    context_fields = {"faction_key", "source_key", "unit_key"}
    for entity in entities:
        if (not isinstance(entity, dict) or not required <= entity.keys()
                or entity.keys() - required - context_fields - {"line"}):
            raise RetrievalFailure("invalid_model_output", "Invalid entity declaration.")
        if any(not isinstance(value, str) or not value.strip() for value in entity.values()):
            raise RetrievalFailure("invalid_model_output", "Entity fields must be non-empty strings.")
        key = entity["key"]
        if not re.fullmatch(r"[a-z][a-z_0-9]{0,31}", key) or key in keys:
            raise RetrievalFailure("invalid_model_output", "Entity keys must be unique identifiers.")
        keys.add(key)
        if entity["kind"] not in ENTITY_TABLES or entity["match"] not in ("name", "id"):
            raise RetrievalFailure("invalid_model_output", "Unsupported entity kind or match mode.")
        # A model may extract a name/ID, but cannot fabricate one as resolved context.
        for field in ("value", "line"):
            if field in entity and not re.search(r"(?<!\w)" + re.escape(entity[field].strip()) + r"(?!\w)",
                                                 question, re.IGNORECASE):
                raise RetrievalFailure("invalid_model_output", "Entity value was not supplied in the question.")
        if "line" in entity and (entity["kind"] != "ability" or "unit_key" not in entity):
            raise RetrievalFailure("invalid_model_output", "A card line requires an ability and unit reference.")
    by_key = {entity["key"]: entity for entity in entities}
    for entity in entities:
        for field in entity.keys() & context_fields:
            kind = field.removesuffix("_key")
            target = by_key.get(entity[field])
            if (kind not in ENTITY_TABLES[entity["kind"]][2]
                    or target is None or target["kind"] != kind):
                raise RetrievalFailure("invalid_model_output", "Invalid entity context reference.")
    return entities


def _normalize_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _match_names(value: str, candidates: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    value = _normalize_name(value)
    by_name = {}
    for candidate in candidates:
        by_name.setdefault(_normalize_name(candidate["name"]), []).append(candidate)
    if value in by_name:
        return by_name[value], "exact"

    phrase = re.compile(r"(?<!\w)" + re.escape(value) + r"(?!\w)")
    matches = [candidate for name, identities in by_name.items() if phrase.search(name)
               for candidate in identities]
    if matches:
        return matches, "phrase"

    # One extra name lets the caller detect that suggestions were capped.
    names = get_close_matches(value, by_name, n=FUZZY_CANDIDATE_LIMIT + 1, cutoff=0.6)
    return [candidate for name in names for candidate in by_name[name]], "fuzzy"


def _lookup_candidates(entity: dict[str, str], path: Path,
                       contexts: dict[str, dict[str, str]]) -> ExecutedQuery:
    table, id_column, context_columns = ENTITY_TABLES[entity["kind"]]
    fields = {"id": f'e."{id_column}"', "name": 'e."name"'}
    joins = []
    from_clause = f'"{table}"'
    match_column = id_column
    if entity["kind"] == "ability":
        scoped_card = "unit" in contexts or "source" in contexts
        card_filter = "" if scoped_card else "WHERE COALESCE(da.name, '') <> ''"
        from_clause = "(" + ABILITY_ENTITIES.format(card_filter=card_filter) + ")"
        fields.update(line='e."line"', ability_id='e."ability_id"')
        if scoped_card:
            match_column = "ability_id"
    predicates = [f'e."{match_column}" = :value' if entity["match"] == "id" else 'e."name" IS NOT NULL']
    args = {"value": entity["value"].strip()} if entity["match"] == "id" else {}
    if "line" in entity:
        predicates.append('e."line" = :line')
        args["line"] = entity["line"].strip()
    for kind, column in context_columns.items():
        context_table, context_id, _ = ENTITY_TABLES[kind]
        fields[f"{kind}_id"] = f'e."{column}"'
        fields[f"{kind}_name"] = f'{kind}."name"'
        joins.append(f'LEFT JOIN "{context_table}" {kind} ON {kind}."{context_id}" = e."{column}"')
        if kind in contexts:
            predicate = f'e."{column}" = :{kind}_id'
            if entity["kind"] == "ability" and kind == "faction":
                predicate = f"({predicate} OR (e.Id IS NOT NULL AND COALESCE(e.factionId, '') = ''))"
            predicates.append(predicate)
            args[f"{kind}_id"] = contexts[kind]["id"]
    projection = ", ".join(f'{value} AS "{name}"' for name, value in fields.items())
    order = ", ".join(f'"{name}"' for name in fields)
    sql = (f'SELECT DISTINCT {projection} FROM {from_clause} e {" ".join(joins)} '
           f'WHERE {" AND ".join(predicates)} ORDER BY {order}')
    # ponytail: scan small name catalogs per lookup; index them if profiling warrants it.
    limits = NAME_LOOKUP_LIMITS if entity["match"] == "name" else ExecutionLimits()
    return execute_bounded(path, sql, args, limits=limits)


def resolve_entities(entities: list[dict[str, str]], path: Path) -> EntityResolution:
    resolved = {}
    scoped_keys = {entity[field] for entity in entities for field in ("faction_key", "source_key", "unit_key")
                   if field in entity}
    priority = {"faction": 0, "source": 0, "unit": 1}
    ordered = sorted(entities, key=lambda e: priority.get(e["kind"], 2))
    for entity in ordered:
        contexts = {}
        for kind in ENTITY_TABLES[entity["kind"]][2]:
            reference = entity.get(f"{kind}_key")
            choices = ([resolved[reference]] if reference else
                       [value for key, value in resolved.items()
                        if value["kind"] == kind and key not in scoped_keys])
            if len(choices) > 1:
                return EntityResolution(status="needs_clarification", resolved=resolved,
                                        candidates=choices,
                                        message=f'Specify which {kind} applies to {entity["value"]}.')
            if choices:
                contexts[kind] = choices[0]
        query = _lookup_candidates(entity, path, contexts)
        if entity["match"] == "name" and query.truncated:
            return EntityResolution(status="needs_clarification", resolved=resolved, truncated=True,
                                    message="Name catalog exceeds the lookup budget; narrow the entity context.")
        candidates = [{"key": entity["key"], "kind": entity["kind"],
                       **{column: value for column, value in zip(query.columns, row)
                          if value not in (None, "")}}
                      for row in query.rows]
        match_type = "id"
        if entity["match"] == "name":
            candidates, match_type = _match_names(entity["value"], candidates)
        for candidate in candidates:
            candidate.update(matched_text=entity["value"], match_type=match_type)
        limit = FUZZY_CANDIDATE_LIMIT if match_type == "fuzzy" else ExecutionLimits().rows
        truncated = query.truncated or len(candidates) > limit
        candidates = candidates[:limit]
        if match_type == "fuzzy" or len(candidates) != 1 or truncated:
            if not candidates:
                message = "No matching name or ID; supply a corrected name or exact ID."
            elif match_type == "fuzzy":
                message = "Possible spelling matches; confirm a candidate identity."
            else:
                message = "Multiple matches; specify a candidate identity or more context."
            return EntityResolution(status="needs_clarification", resolved=resolved,
                                    candidates=candidates, message=message, truncated=truncated)
        resolved[entity["key"]] = candidates[0]
    return EntityResolution(resolved=resolved)


def identity_fields(entity: dict[str, str]) -> tuple[str, ...]:
    if "id" not in entity:
        return ("unit_id", "line")
    return ("id", "faction_id") if entity["kind"] == "ability" else ("id",)


def resolved_bindings(resolved: dict[str, dict[str, str]]) -> dict[str, str]:
    """Preserve catalog faction scopes and the real (DatasheetId, line) card key."""
    bindings = {}
    for key, entity in resolved.items():
        for field in identity_fields(entity):
            name = f"{key}_{field}"
            value = entity.get(field, "")  # Shared catalog abilities have an empty faction.
            if name in bindings and bindings[name] != value:
                raise RetrievalFailure("invalid_model_output", "Entity keys produce conflicting parameter names.")
            bindings[name] = value
    return bindings
