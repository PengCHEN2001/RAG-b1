def get_unit_all_metadata(d_id: str) -> tuple[str, dict[str, str]]:
    """Build SQLite SQL with named parameters, preserving string IDs."""
    if not isinstance(d_id, str) or not d_id.strip():
        raise ValueError("d_id must be a non-empty string")

    sql = """
        SELECT d.Id, d.name, d.factionId, f.name AS faction_name,
        d.role, d.legend, d.link
        FROM datasheets d
        LEFT JOIN factions f ON f.Id = d.factionId
        WHERE d.Id = :d_id
    """
    return sql, {"d_id": d_id}


# Keep model-facing descriptions and application-side callables together.
# Registered builders return (SQLite SQL with :name placeholders, parameter dict).
FUNCTION_REGISTRY = {
    "get_unit_all_metadata": {
        "description": (
            "Get a unit's basic metadata: ID, name, faction, role, lore, and link. "
            "Does not retrieve model stats, weapons, abilities, or points. "
            "Requires an already resolved datasheet ID; never infer an ID from a name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "d_id": {
                    "type": "string",
                    "description": "Exact datasheets.Id, preserving leading zeros.",
                    "minLength": 1,
                }
            },
            "required": ["d_id"],
            "additionalProperties": False,
        },
        "function": get_unit_all_metadata,
    }
}


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
