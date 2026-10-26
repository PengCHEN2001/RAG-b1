# RAG-b1

RAG-b1 is a Retrieval-Augmented Generation project for answering Warhammer 40K questions using a core rulebook PDF and a SQLite unit database.

The implementation includes PDF ingestion, hybrid rulebook/SQL retrieval, a LangGraph agent with session memory, and a CLI with local graph traces.

After `uv sync`, configure `GOOGLE_API_KEY` in `.env` and prepare the database and rulebook indexes as described in the Project Guide. Start an interactive session or ask one question:

```bash
uv run python -m ui.cli --trace
uv run python -m ui.cli "Which abilities does Warboss have?" --trace --sources
```

Commands: `/new`, `/sources [ID]`, `/trace`, `/quit`. Traces are retained by default under `data/traces/<session_id>/`; `--no-trace` disables retention. Read a run without model calls using `uv run python -m ui.cli --inspect path/to/run.jsonl` (add `--json` for full node inputs/outputs).

## Documentation

- See the [Project Guide](docs/project-guide.md) for the project structure, current design, setup, usage, and testing instructions.
- See the [Development Journal](docs/journal.md) for completed work, implementation decisions, and results.
