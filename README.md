# RAG-b1

RAG-b1 is a Warhammer 40K question-answering system built around two knowledge sources: the Core Rules PDF and a SQLite unit database. It includes a hybrid Rulebook Retriever, a safe Text-to-SQL Retriever, a LangGraph agent, CLI and Textual TUI interfaces, and an evaluation pipeline.

## Contents

- [Team](#team)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Tests and Evaluation](#tests-and-evaluation)
- [Technical Overview](#technical-overview)
- [Notes](#notes)
- [Troubleshooting](#troubleshooting)

## Team

Paris Cité Université - M2 Computational Linguistics - 2026-2027

- Jiahua ZHANG — 22415261 - jiahuazh_07@outlook.com
- Peng CHEN — 22409126 - sslchenpeng@gmail.com

## Documentation

- [Project Guide](docs/project-guide.md): system architecture, implementation details, technical choices and their justifications.
- [Evaluation Protocol](docs/eval_protocol.md): datasets, metrics, evaluation workflow and report format.
- [Example Evaluation Report](evaluation/reports/20261028_173548/report.md): one automatically generated benchmark report, including its configuration and failures. It is provided as a reference output, not as a fixed pass threshold.

## Project Structure

```text
RAG-b1/
├── data/                                  # Source data and local generated artifacts
│   ├── pdf/
│   │   └── Core Rules.pdf                  # Source Rulebook PDF
│   ├── sql/
│   │   ├── wahadb.sqlite                   # Warhammer unit database
│   │   └── SQL_AGENT_DATA_DICTIONARY.md    # Database schema and query guidance
│   ├── processed/
│   │   ├── Core Rules.corrections.json     # Versioned visual-to-text correction
│   │   ├── Core Rules.json                 # Structured Docling parse
│   │   ├── Core Rules.md                   # Readable parse preview
│   │   ├── Core Rules.chunks.jsonl         # Canonical Rulebook chunks
│   │   └── images/Core Rules/
│   │       ├── picture-*.png               # Extracted PDF images
│   │       └── manifest.json               # Image page and bounding-box metadata
│   ├── indexes/core_rules/                 # Rebuildable local retrieval indexes
│   │   ├── chroma/                         # Persistent Dense vector index
│   │   ├── bm25.pkl                        # BM25 keyword index
│   │   ├── bm25_chunk_ids.json             # BM25 position-to-chunk mapping
│   │   └── index_manifest.json             # Index version and hash metadata
│   └── traces/                             # Local CLI and Agent execution traces
├── ingestion/                              # Rulebook preparation pipeline
│   ├── pdf_parser.py                       # Docling PDF parsing
│   ├── chunker.py                          # Section-aware chunking and corrections
│   └── build_index.py                      # Embedding, Chroma and BM25 index building
├── retrieval/
│   └── rulebook_retriever.py               # Dense/BM25/RRF/reranker Rulebook retrieval
├── sql_retriever/                          # Fixed and generated SQL retrieval
│   ├── hybrid.py                           # Main SQL retrieval entry point
│   ├── fixed_func.py                       # Fixed-query routing and execution
│   ├── functions.py                        # Registered parameterized SQL functions
│   ├── entities.py                         # Entity matching and clarification
│   ├── constraints.py                      # Generated-SQL entity constraints
│   ├── safety.py                           # Read-only SQL validation and limits
│   └── result.py                           # Structured SQL result type
├── llm/                                    # Gemini configuration, pacing and usage tracking
│   ├── llm_config.py                       # Shared model configuration
│   ├── llm.py                              # Google AI Studio client
│   └── instrumentation.py                  # Retries, pacing and token tracking
├── agent/                                  # LangGraph planning, evidence and response workflow
│   ├── graph.py                            # Agent and evidence graphs
│   ├── nodes.py                            # Planning and retrieval nodes
│   ├── response.py                         # Answer generation and citations
│   ├── state.py                            # Conversation state and memory
│   └── runtime.py                          # Turn execution and trace recording
├── ui/
│   ├── cli.py                              # Command-line chat interface
│   └── tui.py                              # Textual terminal interface
├── evaluation/                             # Benchmarks, metrics and evaluation runner
│   ├── datasets/                           # Frozen Rulebook, SQL and combined datasets
│   ├── adapters/                           # Trace-to-evaluation-record conversion
│   ├── metrics/                            # Retrieval, SQL and Router metrics
│   ├── dataset_loader.py                   # Dataset loading and validation
│   ├── generate_rulebook_testset.py        # RAGAS Rulebook candidate generation
│   ├── finalize_rulebook_testset.py        # Rulebook review and dataset finalisation
│   ├── run.py                              # Unified benchmark command
│   ├── ragas_adapter.py                    # Optional RAGAS judge integration
│   ├── runs/                               # Generated raw outputs; kept local
│   └── reports/
│       └── 20261028_173548/                # Committed example evaluation report
├── tests/                                  # Offline ingestion, retrieval, Agent, UI and evaluation tests
├── docs/
│   ├── project-guide.md                    # Technical design guide
│   └── eval_protocol.md                    # Evaluation design and procedure
├── .env.example                            # Example Google API and runtime configuration
├── .gitignore                              # Generated files and local secrets to exclude
├── Makefile                                # Reproducible preparation and verification commands
├── pyproject.toml                          # Python project and dependency configuration
├── uv.lock                                 # Locked dependency versions
└── README.md                               # Project entry point
```

## Quick Start

Requirements:

- Git;
- [uv](https://docs.astral.sh/uv/);
- GNU Make 4.3 or later;
- a Google AI Studio API key.

```bash
# 1. Clone the project
git clone https://github.com/PengCHEN2001/RAG-b1.git
cd RAG-b1

# 2. Create the local environment file
cp .env.example .env

# 3. Add the Google AI Studio key to .env
# GOOGLE_API_KEY=your_google_ai_api_key

# 4. Install dependencies, process the Rulebook and build the indexes
make prepare

# On macOS with Homebrew GNU Make, use:
gmake prepare
```

`make prepare` is the one-time setup command for a new clone. It installs the locked dependencies, checks the API key configuration and input data, parses the Rulebook when needed, creates the final chunks, applies the documented correction, builds the Chroma and BM25 indexes, validates the benchmark datasets, and runs a local retrieval smoke test.

It prepares everything required before using the chat application or running evaluation. It does not call Gemini, start a chat interface, or run the formal benchmark. The first run requires network access to install dependencies and download the local models. A C++ compiler may be needed if a native Python dependency must be built from source. Unchanged parsing, chunking and indexing steps are skipped on later runs.

## Usage

Start the interactive CLI:

```bash
uv run --env-file .env python -m ui.cli --trace
```

Ask one question and display its sources:

```bash
uv run --env-file .env python -m ui.cli \
  "What is the rule for Grenades?" \
  --trace --sources
```

Start the Textual TUI:

```bash
uv run --env-file .env python -m ui.tui
```

Interactive commands:

```text
/new            Start a new session
/sources [ID]   Display cited sources
/trace          Display the latest run information
/quit           Exit
```

Saved traces are written to `data/traces/<session_id>/<run_id>.jsonl`. To inspect a saved trace without making another model call:

```bash
uv run python -m ui.cli --inspect data/traces/SESSION/RUN.jsonl --sources
uv run python -m ui.tui --inspect data/traces/SESSION/RUN.jsonl
```

Both interfaces support `--model`, `--trace-dir` and `--no-trace`. The CLI also supports `--json` when inspecting the complete node inputs and outputs. The TUI shows chat messages beside live node timings, routes and retrieval counts. `Ctrl+T` toggles the inspector, `Ctrl+S` opens sources, `Enter` opens a selected step, `Esc` requests cancellation, and `Ctrl+Q` exits.

The CLI and TUI report Gemini call counts and provider token usage for each turn and session, including SQL routing, generation and repair calls. Missing provider usage is shown as `null` rather than being treated as zero. `/new` starts a fresh session and resets the displayed session totals.

## Tests and Evaluation

Run the offline test suite before the benchmarks. Each benchmark can then be run separately or together:

```bash
# Run the complete offline test suite
uv run python -m unittest discover -s tests

# Evaluate the six Rulebook cases
uv run --env-file .env python -m evaluation.run --dataset rulebook

# Evaluate the twelve Text-to-SQL cases
uv run --env-file .env python -m evaluation.run --dataset text2sql

# Evaluate the two combined-source cases
uv run --env-file .env python -m evaluation.run --dataset combined

# Run all twenty benchmark cases
uv run --env-file .env python -m evaluation.run --dataset all

# Add optional RAGAS judge metrics to the Rulebook evaluation
uv run --group eval --env-file .env \
  python -m evaluation.run --dataset rulebook --ragas
```

Run the normal evaluation before enabling RAGAS because RAGAS makes additional Gemini requests. Each evaluation creates:

```text
evaluation/runs/<run_id>/raw_results.jsonl
evaluation/reports/<run_id>/summary.json
evaluation/reports/<run_id>/failures.jsonl
evaluation/reports/<run_id>/report.md
```

See the [Evaluation Protocol](docs/eval_protocol.md) for the datasets, metrics and interpretation of failures.

## Technical Overview

The main technical choices are:

- **Docling** for structured PDF parsing with headings, tables, pages and image references;
- **BGE + ChromaDB** for local semantic retrieval;
- **BM25** for exact game terminology and rule names;
- **RRF + Cross-Encoder** for hybrid fusion and reranking;
- **SQLite** with fixed and generated read-only queries for structured data;
- **LangGraph** for routing, retrieval, conversation state and answer generation;
- **Google AI Studio Gemini** as the common LLM backend;
- **RAGAS** for optional Rulebook retrieval and answer-quality metrics.

The implementation and the reasons behind these choices are documented in the [Project Guide](docs/project-guide.md).

## Notes

- The first preparation may download the Docling, embedding and reranker models.
- Google AI Studio free-tier limits can make a complete evaluation slow.
- RAGAS makes additional Gemini calls and should be run separately when needed.
- Conversation state is retained within the current process and is not restored after restarting the application.
- Generated indexes, traces and evaluation runs are not committed by default; one example report is included for reference.

## Troubleshooting

Gemini errors can come from an invalid key, unavailable model, network failure or Google AI Studio quota. An `error at plan` means the Gemini planning request failed before retrieval; the same message at `respond` means retrieval may have completed but answer generation failed. Use `--trace` to locate the failed stage and retain any completed evidence. A `ClientError` alone does not prove that Google servers are unavailable.

The free-tier defaults wait at least five seconds between Gemini SDK calls and retry temporary `429` and `5xx` errors up to three times with bounded exponential backoff. These values can be changed with `GEMINI_MIN_INTERVAL_SECONDS`, `GEMINI_MAX_ATTEMPTS` and `GEMINI_RETRY_BASE_SECONDS`; see `.env.example` for the current defaults.
