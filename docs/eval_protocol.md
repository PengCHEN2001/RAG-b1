# Evaluation Protocol

## Quick start

First configure `GOOGLE_API_KEY` in `.env` or the environment and run
`make prepare` from the repository root. This installs runtime and evaluation
dependencies, builds the rulebook indexes, and validates the existing frozen
datasets. Dataset generation and evaluation are separate from preparation.

Run all 20 cases with the real Agent, but without the additional RAGAS judge:

```bash
uv run --env-file .env python -m evaluation.run --dataset all
```

This command measures Rulebook chunk retrieval, SQL result accuracy, Router
classification, combined-source orchestration, failures, and latency. The Agent
still uses Gemini for routing and answer generation; only the extra RAGAS judge
calls are disabled.

Run the same 20 cases and additionally judge the six Rulebook answers with RAGAS:

```bash
uv run --group eval --env-file .env \
  python -m evaluation.run --dataset all --ragas
```

This adds Context Precision, Context Recall, Faithfulness, Response Relevancy,
and Factual Correctness. It requires the evaluation dependency group and makes
additional Gemini calls.

Each benchmark can also be isolated:

```bash
# Six Rulebook questions: routing plus Hit@5, Precision@5, Recall@5, and MRR.
uv run --env-file .env python -m evaluation.run --dataset rulebook

# The same Rulebook cases plus the five RAGAS judge metrics.
uv run --group eval --env-file .env \
  python -m evaluation.run --dataset rulebook --ragas

# Twelve SQL questions: routing, returned status, columns, rows, errors, and latency.
uv run --env-file .env python -m evaluation.run --dataset text2sql

# Two combined questions: the `both` route and evidence from both data sources.
uv run --env-file .env python -m evaluation.run --dataset combined
```

The runner creates a unique run directory and prints the paths of the generated
raw results, summary, failure list, and Markdown report.

Application calls are paced for the Gemini free tier, so a complete run is
expected to take several minutes. Run the deterministic benchmark first. The
RAGAS integration owns separate LangChain/Gemini clients and is not covered by
the application call limiter; `--ragas` should therefore be run separately and
provider failures must remain visible as evaluation errors.

## Document outline

1. **Purpose** — defines the questions that the evaluation must answer.
2. **Evaluation workflow** — shows how datasets, the Agent, metrics, and reports connect.
3. **Design principles** — explains why the evaluation is layered, project-specific, and reproducible.
4. **Files and responsibilities** — identifies the datasets, evaluation code, source data, and tests.
5. **Benchmark datasets** — describes the 6 Rulebook, 12 Text-to-SQL, and 2 combined cases.
6. **Rulebook evaluation** — explains test-set construction, retrieval metrics, and RAGAS metrics.
7. **Text-to-SQL evaluation** — explains result comparison, status handling, and SQL metrics.
8. **Router and combined evaluation** — defines route scoring and two-source success criteria.
9. **Runtime records and performance** — describes traces, latency, model calls, and token fields.
10. **Execution and outputs** — defines the run sequence and the four generated result files.
11. **Reproducibility and limitations** — records configuration requirements and interpretation limits.

## 1. Purpose

This protocol defines what the project evaluates, how the benchmark data was
constructed, which metrics are reported, and how an evaluation run can be
reproduced. It is designed to answer five questions:

1. Can the Rulebook Retriever find the rule passages required to answer a question?
2. Can the Text-to-SQL component return the correct database result from a natural-language question?
3. Can the Router select the correct source or combination of sources?
4. Can the complete system produce a relevant and grounded answer from the retrieved evidence?
5. What are the system latency, model-call count, and token usage?

The evaluation is intentionally layered. A single final score would not explain
whether a failure came from routing, retrieval, SQL execution, answer generation,
or the evaluation process itself.

## 2. Evaluation workflow

```text
Human-reviewed and frozen benchmark datasets
        │
        ├── rulebook_v1.jsonl
        │       └── Router → Rulebook Retriever → Answer Generation
        │                    ├── deterministic retrieval metrics
        │                    └── optional RAGAS metrics
        │
        ├── text2sql_v1.jsonl
        │       └── Router → SQL Retriever → database result comparison
        │
        └── combined_v1.jsonl
                └── Router → SQL + Rulebook → Answer Generation
                             └── source-use and orchestration checks
        │
        ▼
Trace Adapter → raw results → metrics → failures → summary report
```

Every case runs in an independent session. This prevents the conversation
history or entity memory from one benchmark question from affecting another.
The application still reuses loaded retrieval resources to avoid unnecessary
model and index initialization.

## 3. Design principles

### Layered rather than single-score evaluation

Retrieval, routing, SQL execution, and answer generation are scored separately.
For example, a correct retrieved chunk followed by an incorrect final answer is
an answer-generation failure, not a retrieval failure. This separation makes
the report useful for debugging as well as for final validation.

### Project-specific benchmark data

The project uses its own Rulebook chunks and `wahadb.sqlite` schema. Public
benchmarks can inform the design, but they cannot directly validate the actual
documents, entities, relationships, and failure states used by this application.

### Frozen references and reproducible comparisons

All formal cases are human-reviewed and stored in versioned JSONL files. The
same frozen cases are used across runs. The report records the Git revision and
hashes of the SQLite database, Rulebook chunks, and index manifest so that a
result can be associated with the evaluated project state.

### Failures remain visible

No failed case is silently removed from an average. The runner distinguishes:

- `system_error`: the application or one of its runtime components failed;
- `evaluation_error`: scoring or the RAGAS judge failed;
- `invalid_case`: the benchmark data failed validation.

An evaluation error is not automatically converted into an incorrect system
answer. It is reported separately and retained in the failure file.

## 4. Files and responsibilities

```text
docs/
└── eval_protocol.md                         # Evaluation design and usage

evaluation/
├── datasets/
│   ├── rulebook_v1.jsonl                    # 6 reviewed Rulebook cases
│   ├── rulebook_v1.md                       # Human-readable Rulebook cases
│   ├── text2sql_v1.jsonl                    # 12 reviewed SQL cases
│   ├── text2sql_v1.md                       # Human-readable SQL cases
│   ├── combined_v1.jsonl                    # 2 reviewed two-source cases
│   └── combined_v1.md                       # Human-readable combined cases
├── adapters/
│   └── trace_adapter.py                     # Converts retained Agent traces
├── metrics/
│   └── deterministic.py                     # Rulebook, SQL, and Router metrics
├── dataset_loader.py                        # Loads and validates frozen cases
├── ragas_adapter.py                         # Optional Gemini/RAGAS judge layer
├── run.py                                   # Unified evaluation command
├── runs/<run_id>/raw_results.jsonl          # Complete per-case outputs
└── reports/<run_id>/
    ├── summary.json                         # Machine-readable aggregate metrics
    ├── failures.jsonl                       # Failed cases with actual outputs
    └── report.md                            # Human-readable English report

data/
├── processed/Core Rules.chunks.jsonl        # Chunks used by the Retriever
└── sql/wahadb.sqlite                        # Database used by Text-to-SQL

tests/
├── test_evaluation_trace_adapter.py
├── test_evaluation_deterministic_metrics.py
├── test_evaluation_runner.py
├── test_rulebook_retriever.py
└── test_sql_hybrid.py
```

Generated runs and reports are ignored by default. A selected final report can
be committed explicitly as the project baseline.

## 5. Benchmark datasets

The first frozen version contains 20 cases:

| Dataset | Cases | Main purpose |
| --- | ---: | --- |
| Rulebook | 6 | Retrieval and grounded Rulebook answers |
| Text-to-SQL | 12 | Database results and status handling |
| Combined | 2 | Routing and two-source orchestration |

This is a course-project validation set, not a statistically representative
benchmark. The size and distribution are project design decisions rather than
course-mandated thresholds.

The JSONL files are read by the evaluation code. Matching Markdown files make
the same cases and reference answers easier to inspect manually. A case only
enters the frozen benchmark after its question, reference answer or result,
source information, and expected status have been reviewed.

## 6. Rulebook evaluation

### Test-set construction

`rulebook_v1` contains four `single_hop_specific` cases and two
`multi_hop_abstract` cases. Three cases cover the capabilities explicitly
highlighted for the project—Grenades, Stratagems, and scoring points. Three
additional candidates were produced with RAGAS `TestsetGenerator` from the
same chunks used by the application.

RAGAS generation was used to accelerate candidate creation, not to replace
human judgement. Each generated question was checked to ensure that:

- it can be answered from the Rulebook alone;
- its reference answer is correct and clearly written;
- its reference chunks contain the required evidence;
- its section and page provenance are correct;
- it does not duplicate another case or require outside knowledge.

The evaluation uses the existing chunks instead of re-splitting the PDF. This
ensures that the benchmark measures the knowledge units actually available to
the Retriever.

The evaluated Retriever performs four underlying searches: Dense retrieval on
the Router's semantic rewrite, Dense retrieval on the unchanged user question,
BM25 on the unchanged question, and BM25 on Router keywords. The two Dense
rankings are first fused into one Dense route; this route and the two BM25
routes are then fused and reranked. The hierarchy preserves the original
question when a rewrite is too narrow without giving Dense retrieval twice the
outer voting weight of BM25.

### Deterministic retrieval metrics

Reference and retrieved chunks are matched by `chunk_id`. The following metrics
do not require an LLM judge:

- **Hit@5:** whether at least one reference chunk appears in the first five results;
- **Precision@5:** the proportion of the first five positions occupied by reference chunks;
- **Recall@5:** the proportion of required reference chunks retrieved in the first five results;
- **MRR:** the reciprocal rank of the first retrieved reference chunk.

These metrics answer whether the required evidence was retrieved and where it
appeared in the final ranking. Chunk text, page provenance, retrieval scores,
and rankings remain available in the raw results for failure analysis.

### RAGAS metrics

When `--ragas` is enabled, the retrieved contexts, final response, and reviewed
reference answer are converted to a RAGAS single-turn sample. Gemini is used as
the judge for:

- **Context Precision:** whether useful evidence is ranked ahead of irrelevant evidence;
- **Context Recall:** whether the contexts cover the information in the reference answer;
- **Faithfulness:** whether claims in the response are supported by retrieved evidence;
- **Response Relevancy:** whether the response addresses the question;
- **Factual Correctness:** whether the response agrees with the reviewed reference answer.

The deterministic chunk-ID metrics and the RAGAS scores are both retained.
They measure related but different properties and should not be collapsed into
one overall score.

## 7. Text-to-SQL evaluation

Text-to-SQL uses an execution-result benchmark over the project database rather
than an LLM judge. The 12 reviewed cases cover direct entity queries, attributes,
weapons, abilities, keywords, aggregation, joins, numeric filters, ambiguity,
empty results, and unsupported questions.

Each normal case contains a reviewed `gold_sql`, `expected_columns`, and
`expected_rows`. The gold query establishes and rechecks the expected result;
the generated SQL does not need to use the same syntax. Two different SQL
statements are considered equivalent when they return the same correct result.

The comparison normalizes Unicode text, numeric values, and the known database
encoding issue. It requires the expected result shape and row values, but does
not reject an otherwise equivalent result only because generated SQL uses a
different output alias (for example, `model_name` instead of `unit_name`). Exact
column-name equality is retained as diagnostic information. Row order is ignored
unless the case explicitly sets `order_sensitive`; duplicate-row multiplicity is
preserved.

The reported metrics are:

- **Execution Accuracy:** correct result shape and row values among cases expecting data;
- **Status Accuracy:** correct handling of `ok`, `empty`, `needs_clarification`, and `unsupported`;
- **Result Accuracy:** the complete per-case result, including non-success states;
- **Error Rate:** the proportion of cases returning `error`;
- **Latency:** the measured execution time recorded for each case.

SQL safety is verified separately by unit tests rather than mixed into result
accuracy. The safety tests cover read-only execution, allowed tables, parameter
validation, statement and result limits, time budgets, and rejection of writes,
multiple statements, `ATTACH`, `DETACH`, `PRAGMA`, extension loading, and
unauthorized tables.

## 8. Router and combined evaluation

The Router benchmark reuses the existing questions instead of maintaining a
second duplicate dataset. Expected routes are derived from the dataset purpose:

- Rulebook cases → `rulebook`;
- supported SQL and clarification cases → `sql`;
- combined cases → `both`;
- database-out-of-scope cases → `unsupported`.

Router performance is reported using Accuracy, Macro-F1, per-route Precision
and Recall, and a confusion matrix.

A combined case is successful only when the Router selects `both`, both
Retrievers are actually called, the SQL result matches its referenced SQL case,
all required Rulebook reference chunks are retrieved, the completed answer cites
both source types, and the run outcome is `completed`. This prevents a plausible
answer based on only one source, or a failed SQL object that merely exists, from
being counted as successful. Because the combined set contains only two cases,
their final answer correctness remains marked for direct human inspection in
the final report rather than introducing another small custom LLM judge.

## 9. Runtime records and performance

Trace generation belongs to the application runtime, not to the evaluation
layer. The existing Agent runtime retains one structured trace for each case.
The evaluation code only reads that trace through `trace_adapter.py` and converts
the existing fields into a consistent evaluation record. It does not redefine
the Agent's internal result objects or duplicate its tracing logic.

From the trace, the adapter extracts the user question, selected route, SQL
result, ranked Rulebook chunks, final answer, citations, outcome, errors, total
latency, and node timings. The trace contains structured execution events, not
private model reasoning.

The runtime record also reserves fields for model-call count, input tokens,
output tokens, and first-token latency. When a provider or application layer
does not expose a value, the field is stored as `null`; the evaluator never
invents a zero. This makes the difference between “not measured” and “measured
as zero” explicit in both raw results and reports.

## 10. Execution and outputs

For every formal run, the runner:

1. loads and validates the frozen dataset;
2. records the project and data versions;
3. executes every case in an isolated session;
4. retains the complete raw result;
5. computes deterministic metrics;
6. optionally runs RAGAS;
7. saves all failures;
8. writes machine-readable and human-readable summaries.

The four output files serve different purposes:

- `raw_results.jsonl` preserves every case and its actual system output;
- `summary.json` contains aggregate metrics and version metadata;
- `failures.jsonl` supports debugging without searching the complete run;
- `report.md` presents the configuration, metrics, and failures for review.

No arbitrary pass threshold is set before observing the first complete run.
The first reviewed full run with no unresolved `system_error`,
`evaluation_error`, or `invalid_case` becomes the baseline. A run interrupted
by provider quota or availability is retained for diagnosis but is not promoted
to the project baseline. Later runs can be compared against that baseline to
detect regressions.

## 11. Reproducibility and limitations

The API key is read from `GOOGLE_API_KEY`. The application model, RAGAS judge,
and embedding model can be selected through `GEMINI_MODEL`, `RAGAS_MODEL`, and
`RAGAS_EMBEDDING_MODEL`. Secrets are never written to result files.
Application calls are paced by `GEMINI_MIN_INTERVAL_SECONDS` and temporary
provider failures use bounded exponential retries. Waiting and retry events are
retained in the trace so provider instability is not silently hidden.

Before producing the final report, the complete offline test suite is run with:

```bash
uv run python -m unittest discover -s tests
```

The formal baseline should be run from a committed working tree. The report's
`git_commit` and `git_dirty` fields make local uncommitted changes explicit; a
dirty exploratory run is useful for debugging but is not the final reproducible
result.

The evaluation has the following limitations:

- the benchmark is deliberately small and supports project validation rather than statistical claims;
- synthetic Rulebook candidates may retain generation bias despite human review;
- LLM-judge metrics may vary slightly between runs or judge-model versions;
- Text-to-SQL is evaluated against one frozen SQLite snapshot;
- an incorrect SQL query can occasionally return the correct result on a specific snapshot;
- Rulebook scores depend on the quality of the manually selected reference chunks;
- unavailable token or model-usage metadata is reported as `null` rather than estimated.
- the frozen benchmark is single-turn and does not score conversational memory;
  follow-ups, clarification continuation, topic switching and `/new` are checked
  separately through offline tests and manual CLI acceptance;
- RAGAS judge calls currently use their own client and are not paced by the
  application's Gemini throttle.

For these reasons, the final assessment combines deterministic retrieval
metrics, database-result accuracy, RAGAS answer metrics, and direct examination
of failure cases. No single aggregate number is treated as a complete measure
of system quality.

### Final evaluation reminder

Before selecting the final report, run the offline tests and deterministic
benchmarks from a committed working tree, confirm that no provider failure is
being misreported as a model-quality failure, manually review the two combined
answers, and record the separate multi-turn chat checks. Run the optional RAGAS
judge after the deterministic run when Gemini quota is available. Keep failed
cases in the report; do not select a quota-interrupted run as the baseline.
