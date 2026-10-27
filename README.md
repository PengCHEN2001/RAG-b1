# RAG-b1

RAG-b1 is a Retrieval-Augmented Generation project for answering Warhammer 40K questions using a core rulebook PDF and a SQLite unit database.

The implementation includes PDF ingestion, hybrid rulebook/SQL retrieval, a LangGraph agent with session memory, and CLI / Textual TUI interfaces with local graph traces.

After `uv sync`, configure `GOOGLE_API_KEY` in `.env` and prepare the database and rulebook indexes as described in the Project Guide. Start an interactive session or ask one question:

```bash
uv run python -m ui.tui
uv run python -m ui.cli --trace
uv run python -m ui.cli "Which abilities does Warboss have?" --trace --sources
```

Commands: `/new`, `/sources [ID]`, `/trace`, `/quit`. Traces are retained by default under `data/traces/<session_id>/`; `--no-trace` disables retention. Read a run without model calls using `uv run python -m ui.cli --inspect path/to/run.jsonl` (add `--json` for full node inputs/outputs).

The TUI shows chat alongside live node timings, routes and retrieval counts. Select a turn to inspect it; press Enter on a step for its input/output, or open Sources for cited SQL rows and rulebook chunks. `Ctrl+T` toggles the inspector (a separate view on narrow terminals), `Ctrl+S` opens sources, `Esc` requests cancellation, and `Ctrl+Q` quits. Cancellation/quit waits for an in-flight synchronous call to return; new input stays disabled until it stops. Answers appear when the turn finishes.

Open a retained run in the same TUI without API calls: `uv run python -m ui.tui --inspect data/traces/SESSION/RUN.jsonl`. Both interfaces support `--model`, `--trace-dir` and `--no-trace`. Disabling trace files leaves only the current turn available for inspection.

CLI/TUI report LLM SDK calls and provider token usage per turn and session, including SQL routing, generation and repair. API errors and missing usage remain `null`; incomplete totals retain separate known subtotals and coverage counts. `/new` resets session totals, and inspecting an old turn does not count it again. Token reporting works with `--no-trace`; older traces show unavailable usage.

## Documentation

- See the [Project Guide](docs/project-guide.md) for the project structure, current design, setup, usage, and testing instructions.
- See the [Development Journal](docs/journal.md) for completed work, implementation decisions, and results.
