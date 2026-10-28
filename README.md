# RAG-b1

RAG-b1 is a Retrieval-Augmented Generation project for answering Warhammer 40K questions using a core rulebook PDF and a SQLite unit database.

The implementation includes PDF ingestion, hybrid rulebook/SQL retrieval, a LangGraph agent with session memory, and CLI / Textual TUI interfaces with local graph traces.

Install `uv` and GNU Make 4.3 or newer (`gmake` on macOS when installed separately). From the repository root, copy `.env.example` to `.env`, configure `GOOGLE_API_KEY`, then prepare the project with one command:

```bash
make prepare
```

This uses uv-managed Python and installs the runtime and evaluation dependencies, parses the PDF, applies chunk corrections, builds Chroma/BM25 indexes, validates the committed benchmark datasets, and runs a local retrieval check to cache the embedding and reranker models. The first run needs network access for Python, dependencies, and model downloads; a C++ compiler is needed when native dependencies build from source. Repeated runs skip unchanged parsing, chunking, and indexing. Preparation checks the local key configuration without calling Gemini; evaluation and dataset generation are run separately.

Start an interactive session, ask one question, or run the existing evaluation:

```bash
uv run python -m ui.tui
uv run python -m ui.cli --trace
uv run python -m ui.cli "Which abilities does Warboss have?" --trace --sources
uv run --group eval python -m evaluation.run --dataset all --ragas
```

Commands: `/new`, `/sources [ID]`, `/trace`, `/quit`. Traces are retained by default under `data/traces/<session_id>/`; `--no-trace` disables retention. Read a run without model calls using `uv run python -m ui.cli --inspect path/to/run.jsonl` (add `--json` for full node inputs/outputs).

The TUI shows chat alongside live node timings, routes and retrieval counts. Select a turn to inspect it; press Enter on a step for its input/output, or open Sources for cited SQL rows and rulebook chunks. `Ctrl+T` toggles the inspector (a separate view on narrow terminals), `Ctrl+S` opens sources, `Esc` requests cancellation, and `Ctrl+Q` quits. Cancellation/quit waits for an in-flight synchronous call to return; new input stays disabled until it stops. Answers appear when the turn finishes.

Open a retained run in the same TUI without API calls: `uv run python -m ui.tui --inspect data/traces/SESSION/RUN.jsonl`. Both interfaces support `--model`, `--trace-dir` and `--no-trace`. Disabling trace files leaves only the current turn available for inspection.

CLI/TUI report LLM SDK calls and provider token usage per turn and session, including SQL routing, generation and repair. API errors and missing usage remain `null`; incomplete totals retain separate known subtotals and coverage counts. `/new` resets session totals, and inspecting an old turn does not count it again. Token reporting works with `--no-trace`; older traces show unavailable usage.

## Troubleshooting

If the CLI reports `error at plan: ClientError. Check the trace for stack/validation locations; check credentials/network for model failures.`, the planning request was rejected by the Google Gemini API before retrieval started. The same error at `respond` means the answer-generation request failed; any completed retrieval remains available in the trace. Check that `GOOGLE_API_KEY` is valid, the configured Gemini model is available to that key, the network connection works, and the Google AI Studio quota has not been exceeded. A `ClientError` can represent an invalid request, authentication or permission failure, or rate/quota limit; it does not by itself prove a Google server outage. Temporary API failures can be retried later.

## Documentation

- See the [Project Guide](docs/project-guide.md) for the project structure, current design, setup, usage, and testing instructions.
- See the [Development Journal](docs/journal.md) for completed work, implementation decisions, and results.
