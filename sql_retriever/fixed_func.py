import json
import sqlite3
import sys

from pathlib import Path
from contextlib import closing

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sql_retriver.functions import FUNCTION_REGISTRY, get_function_declarations
else:
    from .functions import FUNCTION_REGISTRY, get_function_declarations

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sql" / "wahadb.sqlite"

FIXED_FUNC_SYSTEM_PROMPT = """Select at most one registered function for the request.
Use only the functions and arguments declared below. Do not generate SQL.
Use a datasheet ID only if explicitly supplied or resolved in the input context.
Never invent IDs, translate names into guessed IDs, or remove leading zeros.
If no function fits, or a required ID is missing, return
{"name": null, "arguments": {}}. This means the request needs another tool
or entity resolution before a database query can be built.

Available functions:
""" + json.dumps(get_function_declarations(), ensure_ascii=False, indent=2)


response_schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": ["string", "null"],
            "enum": [*FUNCTION_REGISTRY, None],
            "description": "Registered function name, or null if no call can be made.",
        },
        "arguments": {
            "description": "Arguments matching the selected function; empty if name is null.",
            "anyOf": [
                spec["parameters"] for spec in FUNCTION_REGISTRY.values()
            ] + [{"type": "object", "properties": {}, "additionalProperties": False}],
        }
    },
    "required": ["name", "arguments"],
    "additionalProperties": False,
}



def parse_function_call(response_text: str) -> tuple[str, dict[str, str]] | None:
    """Validate a model selection, then build SQL without executing it."""

    selection = json.loads(response_text)

    if not isinstance(selection, dict) or set(selection) != {"name", "arguments"}:
        raise ValueError("Expected exactly name and arguments")

    name = selection["name"]
    arguments = selection["arguments"]

    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    if name is None:
        if arguments:
            raise ValueError("A null function name requires empty arguments")
        return None
    if not isinstance(name, str) or name not in FUNCTION_REGISTRY:
        raise ValueError("Unknown function name")

    spec = FUNCTION_REGISTRY[name]
    parameters = spec["parameters"]


    if set(parameters["required"]) - arguments.keys():
        raise ValueError("Missing required function arguments")
    if arguments.keys() - parameters["properties"].keys():
        raise ValueError("Unexpected function arguments")
    # Current registered parameters are strings. Extend this validation when
    # introducing other parameter types; schema adherence alone is insufficient.
    if any(not isinstance(value, str) or not value.strip() for value in arguments.values()):
        raise ValueError("Function arguments must be non-empty strings")

    return spec["function"](**arguments)


def make_sql_query(sql: str, args: dict[str, str]) -> list[tuple]:
    """Execute a registered SQLite query with named bindings in read-only mode."""
    db = DB_PATH

    with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as conn:
        rows = conn.execute(sql, args).fetchall()

    return rows


def sql_retrieve_fixed_funcs(query: str, *, client=None) -> list[tuple] | None:
    """Select and execute a fixed query; return rows, or None if no tool fits."""
    if client is None:
        from llm.llm import llm_client

        client = llm_client

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=query,
        config={
            "system_instruction": FIXED_FUNC_SYSTEM_PROMPT,
            "response_mime_type": "application/json",
            "response_json_schema": response_schema,
            # The application validates and dispatches the returned JSON itself.
            "automatic_function_calling": {"disable": True},
        },
    )

    if not response.text:
        raise ValueError("Model returned no JSON text")

    sql_query = parse_function_call(response.text)
    if sql_query is None:
        return None

    sql, args = sql_query
    return make_sql_query(sql, args)
