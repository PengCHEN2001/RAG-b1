"""Live schema grounding plus reviewed rules-reference semantics."""

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from .result import RetrievalFailure

TABLE_PURPOSES = {
    "abilities": "Faction ability catalog; IDs may repeat across factions.",
    "datasheets": "Unit cards; role may be empty. String IDs preserve leading zeros.",
    "datasheets_abilities": "Card ability text overrides non-empty catalog fields.",
    "datasheets_detachment_abilities": "Stored unit-to-detachment ability links.",
    "datasheets_enhancements": "Stored unit-to-enhancement links.",
    "datasheets_keywords": "Unit/model keywords and faction-keyword flags.",
    "datasheets_leader": "LeaderId leads attachedId; both refer to datasheets.",
    "datasheets_models": "Model stat lines, stored as TEXT including symbolic values.",
    "datasheets_models_cost": "Cost per description/configuration, not a universal unit price.",
    "datasheets_options": "Unit equipment options and conditions as text.",
    "datasheets_stratagems": "Stored unit-to-stratagem links; not game-state eligibility.",
    "datasheets_unit_composition": "Unit composition descriptions.",
    "datasheets_wargear": "Weapon profiles; dice, saves, and ranges may be symbolic TEXT.",
    "detachment_abilities": "Detachment ability rule text.",
    "detachments": "Faction detachment catalog.",
    "detachments_chapter_dp": "Detachment chapter/DP mapping.",
    "enhancements": "Faction/detachment enhancements, costs and rules.",
    "factions": "Faction IDs and names.",
    "last_update": "Database refresh timestamp.",
    "source": "Publication metadata and errata references.",
    "stratagems": "Faction/detachment stratagem rules and CP costs.",
}

# Each edge is an equality between exact string keys, not a declared foreign key.
JOIN_EDGES = [
    ("factions", "Id", "datasheets", "factionId"),
    ("factions", "Id", "abilities", "factionId"),
    ("factions", "Id", "detachments", "factionId"),
    ("factions", "Id", "detachment_abilities", "factionId"),
    ("factions", "Id", "stratagems", "FactionId"),
    ("factions", "Id", "enhancements", "FactionId"),
    *[("datasheets", "Id", table, "DatasheetId") for table in (
        "datasheets_abilities", "datasheets_detachment_abilities",
        "datasheets_enhancements", "datasheets_keywords", "datasheets_models",
        "datasheets_models_cost", "datasheets_options", "datasheets_stratagems",
        "datasheets_unit_composition", "datasheets_wargear",
    )],
    ("datasheets", "Id", "datasheets_leader", "LeaderId"),
    ("datasheets", "Id", "datasheets_leader", "attachedId"),
    ("abilities", "Id", "datasheets_abilities", "abilityId"),
    ("detachment_abilities", "Id", "datasheets_detachment_abilities", "detachmentAbilityId"),
    ("enhancements", "id", "datasheets_enhancements", "enhancementId"),
    ("stratagems", "id", "datasheets_stratagems", "stratagemId"),
    ("detachments", "Id", "detachment_abilities", "detachmentId"),
    ("detachments", "Id", "stratagems", "detachmentId"),
    ("detachments", "Id", "enhancements", "detachmentId"),
    ("detachments", "Id", "detachments_chapter_dp", "DetachmentId"),
    ("source", "Id", "datasheets", "sourceId"),
]


def integer_expression(column_sql: str) -> str:
    """Reviewed conversion: only plain signed integers of at most 18 characters."""
    value = f"TRIM({column_sql})"
    return (
        f"CASE WHEN LENGTH({value}) <= 18 AND "
        f"(({value} GLOB '[0-9]*' AND NOT {value} GLOB '*[^0-9]*') OR "
        f"({value} GLOB '-[0-9]*' AND NOT {value} GLOB '-*[^0-9]*')) "
        f"THEN CAST({value} AS INTEGER) ELSE NULL END"
    )


SEMANTIC_GUIDANCE = """Use SQLite, exact string IDs, explicit columns, named :parameters
for all user-supplied values, and only the documented join-key equalities.
Use EXISTS for membership filters and COUNT(DISTINCT unit_id) when links duplicate.
Prefer LEFT JOIN to preserve missing metadata; never equate empty role with keyword.
Card-specific non-empty name/description override the ability catalog using COALESCE
and NULLIF. Catalog ability IDs can repeat: account for faction and duplicate rows.
Points belong to a cost description/configuration. Return that description.
Never CAST raw TEXT statistics or compare/order them numerically without the supplied
integer CASE template. The template returns NULL for dice, saves such as 2+, empty
strings and non-integer statistics. SUM/AVG also require guarded numeric inputs.
Do not infer game-state eligibility, army legality, expected damage, or missing rules.
Clarify undefined 'best'/'cheapest', ambiguous singular names, or unspecified cost
configurations. No query can certify that a stored association is currently legal.
Only SELECT and non-recursive CTEs; explicit INNER/LEFT joins, subqueries, aggregates.
No SELECT *, except COUNT(*); no UNION, windows, arithmetic or cross/implicit joins.
At most four joins, three nested SELECT levels, 16 KiB SQL, and literal LIMIT/OFFSET.
Functions: COUNT SUM AVG MIN MAX COALESCE NULLIF LOWER UPPER TRIM LENGTH ABS ROUND,
LIKE and GLOB. Only the reviewed integer CASE may contain CAST or numeric literals
outside LIMIT/OFFSET. Empty string in NULLIF and integer-template literals are allowed.
Use :parameters even for constant projected answers. Return facts, not explanations.
"""


def open_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.1)
    try:
        connection.enable_load_extension(False)
        connection.execute("PRAGMA query_only = ON")
    except Exception:
        connection.close()
        raise
    return connection


@dataclass
class SchemaContext:
    tables: dict[str, dict[str, str]]

    @property
    def edges(self) -> set[frozenset[tuple[str, str]]]:
        return {
            frozenset(((a.lower(), ac.lower()), (b.lower(), bc.lower())))
            for a, ac, b, bc in JOIN_EDGES
            if a in self.tables and b in self.tables
            and ac.lower() in {c.lower() for c in self.tables[a]}
            and bc.lower() in {c.lower() for c in self.tables[b]}
        }

    def prompt(self) -> str:
        catalog = {name: {"purpose": TABLE_PURPOSES[name], "columns": columns}
                   for name, columns in self.tables.items()}
        edges = [edge for edge in JOIN_EDGES
                 if frozenset(((edge[0].lower(), edge[1].lower()),
                               (edge[2].lower(), edge[3].lower()))) in self.edges]
        return (SEMANTIC_GUIDANCE + "\nSchema:\n" + json.dumps(catalog) +
                "\nJoin edges:\n" + json.dumps(edges) +
                "\nInteger template (substitute a real qualified column for d.value):\n" +
                integer_expression("d.value"))


def inspect_schema(path: Path) -> SchemaContext:
    """Never expose unknown tables, views, or internal SQLite objects."""
    with closing(open_readonly(path)) as connection:
        found = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        tables = {}
        for name in TABLE_PURPOSES:
            if name in found:
                tables[name] = {row[1]: row[2] or "TEXT" for row in connection.execute(
                    f'PRAGMA table_info("{name}")')}
    if not tables:
        raise RetrievalFailure("schema_unavailable", "No approved application tables were found.")
    return SchemaContext(tables)
