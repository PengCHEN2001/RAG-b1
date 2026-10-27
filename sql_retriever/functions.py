def get_unit_all_metadata(d_id: str) -> tuple[str, dict[str, str]]:
    """Build SQLite SQL with named parameters, preserving string IDs."""
    sql = """
        SELECT d.Id, d.name, d.factionId, f.name AS faction_name,
        d.role, d.legend, d.link
        FROM datasheets d
        LEFT JOIN factions f ON f.Id = d.factionId
        WHERE d.Id = :d_id
    """
    return sql, {"d_id": d_id}


def get_unit_abilities(d_id: str) -> tuple[str, dict[str, str]]:
    """Get catalog abilities and rules stored directly on the unit card."""
    sql = """
        SELECT DISTINCT d.name AS datasheet_name,
        NULLIF(da.abilityId, '') AS ability_id,
        COALESCE(NULLIF(da.name, ''), a.name) AS ability_name,
        COALESCE(NULLIF(da.description, ''), a.description) AS ability_text
        FROM datasheets d
        LEFT JOIN datasheets_abilities da ON da.DatasheetId = d.Id
        LEFT JOIN abilities a ON a.Id = da.abilityId
            AND (a.factionId = d.factionId OR COALESCE(a.factionId, '') = '')
        WHERE d.Id = :d_id
    """
    return sql, {"d_id": d_id}


def get_unit_model_stats(d_id: str) -> tuple[str, dict[str, str]]:
    """Return every model profile, preserving symbolic statistics as stored strings."""
    sql = """
        SELECT d.Id AS datasheet_id, d.name AS datasheet_name,
        m.line AS model_line, m.name AS model_name,
        m.m AS movement, m.t AS toughness, m.sv AS save,
        m.invSv AS invulnerable_save, m.invSvDescr AS invulnerable_save_description, m.w AS wounds,
        m.ld AS leadership, m.oc AS objective_control
        FROM datasheets d
        JOIN datasheets_models m ON m.DatasheetId = d.Id
        WHERE d.Id = :d_id
        ORDER BY m.line
    """
    return sql, {"d_id": d_id}


def get_unit_keywords(d_id: str) -> tuple[str, dict[str, str]]:
    """Get keywords, model context, and faction-keyword flags."""
    sql = """
        SELECT d.name AS datasheet_name,
        dk.keyword, dk.model, dk.isFactionKeyword
        FROM datasheets d
        LEFT JOIN datasheets_keywords dk ON dk.DatasheetId = d.Id
        WHERE d.Id = :d_id
    """
    return sql, {"d_id": d_id}


def get_unit_stratagems(d_id: str) -> tuple[str, dict[str, str]]:
    """Get stratagems linked to a unit in the database."""
    sql = """
        SELECT d.name AS datasheet_name,
        s.id AS stratagem_id, s.name AS stratagem_name,
        s.type, s.cpCost, s.phase, s.description
        FROM datasheets d
        LEFT JOIN datasheets_stratagems ds ON ds.DatasheetId = d.Id
        LEFT JOIN stratagems s ON s.id = ds.stratagemId
        WHERE d.Id = :d_id
    """
    return sql, {"d_id": d_id}


def get_unit_weapons(d_id: str) -> tuple[str, dict[str, str]]:
    """Get weapon profiles with their original string-valued statistics."""
    sql = """
        SELECT d.name AS datasheet_name,
        dw.name AS weapon_name, dw.type, dw."range",
        dw.a, dw.bsWs, dw.s, dw.ap, dw.d
        FROM datasheets d
        LEFT JOIN datasheets_wargear dw ON dw.DatasheetId = d.Id
        WHERE d.Id = :d_id
    """
    return sql, {"d_id": d_id}


def get_units_by_faction(
    faction_id: str, role: str | None = None,
) -> tuple[str, dict[str, str]]:
    """List faction units, optionally filtering by an exact stored role."""
    sql = """
        SELECT d.Id, d.name, d.role
        FROM datasheets d
        WHERE d.factionId = :faction_id
    """
    parameters = {"faction_id": faction_id}
    if role is not None:
        sql += " AND d.role = :role"
        parameters["role"] = role
    sql += " ORDER BY d.name"
    return sql, parameters


def get_detachments_and_abilities(faction_id: str) -> tuple[str, dict[str, str]]:
    """List faction detachments and their associated ability IDs and names."""
    sql = """
        SELECT dt.Id AS detachment_id, dt.name AS detachment_name,
        da.Id AS ability_id, da.name AS ability_name
        FROM detachments dt
        LEFT JOIN detachment_abilities da ON da.detachmentId = dt.Id
        WHERE dt.factionId = :faction_id
    """
    return sql, {"faction_id": faction_id}


def _unit_parameters() -> dict:
    return {
        "type": "object",
        "properties": {
            "d_id": {
                "type": "string",
                "description": "Exact datasheets.Id, preserving leading zeros; must already be resolved.",
                "minLength": 1,
            }
        },
        "required": ["d_id"],
        "additionalProperties": False,
    }


def _faction_parameters() -> dict:
    return {
        "type": "object",
        "properties": {
            "faction_id": {
                "type": "string",
                "description": "Exact factions.Id (e.g. TYR), explicitly supplied or already resolved.",
                "minLength": 1,
            }
        },
        "required": ["faction_id"],
        "additionalProperties": False,
    }


def _faction_and_role_parameters() -> dict:
    parameters = _faction_parameters()
    parameters["properties"]["role"] = {
        "type": "string",
        "description": (
            "Optional exact datasheets.role value. Omit if unspecified. "
            "The current database has empty role values; do not infer a role from keywords."
        ),
        "minLength": 1,
    }
    return parameters


# Keep model-facing descriptions and application-side callables together.
# Registered builders return (SQLite SQL with :name placeholders, parameter dict).
FUNCTION_REGISTRY = {
    "get_unit_all_metadata": {
        "description": (
            "Get a unit's basic metadata: ID, name, faction, role, lore, and link. "
            "Does not retrieve model stats, weapons, abilities, or points. "
            "Requires an already resolved datasheet ID; never infer an ID from a name."
        ),
        "parameters": _unit_parameters(),
        "function": get_unit_all_metadata,
    },
    "get_unit_abilities": {
        "description": "Get a unit's ability IDs, names, and rule text, including card-specific abilities.",
        "parameters": _unit_parameters(),
        "function": get_unit_abilities,
    },
    "get_unit_model_stats": {
        "description": (
            "Get all model profiles for one unit: movement (M), Toughness (T), save (Sv), "
            "invulnerable save and its conditions, wounds (W), Leadership (Ld), and Objective Control (OC). "
            "Use for a single characteristic such as Toughness or a full stat line. "
            "Returns each model profile separately with the original string values. "
            "Does not retrieve weapons, abilities, points or compare/rank units."
        ),
        "parameters": _unit_parameters(),
        "function": get_unit_model_stats,
    },
    "get_unit_keywords": {
        "description": "Get a unit's keywords, their model context, and faction-keyword flags.",
        "parameters": _unit_parameters(),
        "function": get_unit_keywords,
    },
    "get_unit_stratagems": {
        "description": (
            "Get stratagems linked to a unit: IDs, names, type, CP cost, phase, and rules. "
            "Database links alone do not establish eligibility in a particular game state."
        ),
        "parameters": _unit_parameters(),
        "function": get_unit_stratagems,
    },
    "get_unit_weapons": {
        "description": "Get a unit's weapon names, types, ranges, attacks, BS/WS, strength, AP, and damage.",
        "parameters": _unit_parameters(),
        "function": get_unit_weapons,
    },
    "get_units_by_faction": {
        "description": "List unit IDs, names, and roles for a faction, sorted by name, with an optional role filter.",
        "parameters": _faction_and_role_parameters(),
        "function": get_units_by_faction,
    },
    "get_detachments_and_abilities": {
        "description": "List a faction's detachment IDs and names with associated ability IDs and names, without rule text.",
        "parameters": _faction_parameters(),
        "function": get_detachments_and_abilities,
    },
}


def build_fixed_query(name: str, arguments: object) -> tuple[str, dict[str, str]]:
    """Validate model-supplied arguments once, then call a trusted SQL builder."""
    if not isinstance(name, str) or name not in FUNCTION_REGISTRY:
        raise ValueError("Unknown function name")
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    spec = FUNCTION_REGISTRY[name]
    parameters = spec["parameters"]
    if set(parameters["required"]) - arguments.keys():
        raise ValueError("Missing required function arguments")
    if arguments.keys() - parameters["properties"].keys():
        raise ValueError("Unexpected function arguments")
    if any(not isinstance(value, str) or not value.strip() for value in arguments.values()):
        raise ValueError("Function arguments must be non-empty strings")
    return spec["function"](**arguments)


def get_function_declarations() -> list[dict]:
    """Return serializable declarations without exposing Python callables."""
    return [
        {
            "name": name,
            "description": spec["description"],
            "parameters": spec["parameters"],
        }
        for name, spec in FUNCTION_REGISTRY.items()
    ]
