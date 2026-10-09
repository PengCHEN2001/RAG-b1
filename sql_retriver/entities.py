"""Resolve explicitly mentioned entities without model-chosen database IDs."""

import re
from pathlib import Path

from .result import RetrievalFailure, RetrievalResult
from .safety import execute_bounded
from .schema import SchemaContext

ENTITY_TABLES = {
    "unit": ("datasheets", "Id", "factionId"),
    "faction": ("factions", "Id", None),
    "detachment": ("detachments", "Id", "factionId"),
    "ability": ("abilities", "Id", "factionId"),
    "enhancement": ("enhancements", "id", "FactionId"),
    "stratagem": ("stratagems", "id", "FactionId"),
    "source": ("source", "Id", None),
}


def validate_entities(entities: object, question: str) -> list[dict[str, str]]:
    if not isinstance(entities, list) or len(entities) > 8:
        raise RetrievalFailure("invalid_model_output", "Entities must be a bounded list.")
    keys = set()
    for entity in entities:
        if not isinstance(entity, dict) or set(entity) != {"key", "kind", "match", "value"}:
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
        value = entity["value"].strip()
        if not re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", question, re.IGNORECASE):
            raise RetrievalFailure("invalid_model_output", "Entity value was not supplied in the question.")
    return entities


def resolve_entities(entities: list[dict[str, str]], path: Path,
                     schema: SchemaContext) -> dict[str, dict[str, str]] | RetrievalResult:
    resolved = {}
    ordered = sorted(entities, key=lambda e: e["kind"] != "faction")
    for entity in ordered:
        table, id_column, faction_column = ENTITY_TABLES[entity["kind"]]
        if table not in schema.tables:
            return RetrievalResult("unsupported", message="Entity table is absent from this database.")
        predicate = (f'"{id_column}" = :value' if entity["match"] == "id" else
                     '"name" = :value COLLATE NOCASE')
        fields = f'"{id_column}", "name"'
        if faction_column:
            fields += f', "{faction_column}"'
        args = {"value": entity["value"].strip()}
        factions = [e["id"] for e in resolved.values() if e["kind"] == "faction"]
        if faction_column and len(factions) == 1:
            predicate += f' AND "{faction_column}" = :faction'
            args["faction"] = factions[0]
        query = execute_bounded(path, f'SELECT DISTINCT {fields} FROM "{table}" '
                                f'WHERE {predicate} ORDER BY "{id_column}"', args, schema)
        candidates = [{"key": entity["key"], "kind": entity["kind"],
                       "id": row[0], "name": row[1],
                       **({"faction_id": row[2]} if faction_column else {})}
                      for row in query.rows]
        if len(candidates) != 1 or query.truncated:
            message = ("No exact match; supply a corrected name or ID." if not candidates else
                       "Multiple exact matches; supply an ID or a faction to disambiguate.")
            return RetrievalResult("needs_clarification", message=message, candidates=candidates,
                                   truncated=query.truncated)
        resolved[entity["key"]] = candidates[0]
    return resolved
