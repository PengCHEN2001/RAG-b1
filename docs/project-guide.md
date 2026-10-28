# Project Guide

## Overview

RAG-b1 is a Warhammer 40K question-answering project using a core rulebook PDF and SQLite unit data through a LangGraph agent, with CLI / Textual TUI interfaces and local graph traces.

## Contents

- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [PDF Processing Design](#pdf-processing-design)
- [Chunking Design](#chunking-design)
- [Embedding Design](#embedding-design)
- [Dense and Keyword Index Design](#dense-and-keyword-index-design)
- [Reproducible Rulebook Build](#reproducible-rulebook-build)
- [Rulebook Retriever Design](#rulebook-retriever-design)
- [SQL Retriever Design](#sql-retriever-design)
- [Agent Graphs and Conversation Replies](#agent-graphs-and-conversation-replies)
- [CLI and Local Graph Traces](#cli-and-local-graph-traces)
- [Build and Simple Verification](#build-and-simple-verification)
- [Evaluation](#evaluation)
- [Final Submission Checklist](#final-submission-checklist)

## System Architecture

```text
User question
      │
      ▼
Planner / Router (Gemini)
      │
      ├── Rulebook question ──► Rulebook Retriever
      ├── Database question ──► SQL Retriever
      ├── Combined question ──► SQL Retriever + Rulebook Retriever
      └── Direct, clarification or unsupported request
      │
      ▼
Collected evidence and source status
      │
      ▼
Answer generation with citations
      │
      ▼
CLI / TUI response and structured trace
```

The following sections describe the Rulebook pipeline, SQL Retriever and Agent workflow in more detail.

## Project Structure

```text
RAG-b1/
├── data/
│   ├── pdf/
│   │   └── Core Rules.pdf
│   ├── sql/
│   │   ├── wahadb.sqlite
│   │   └── SQL_AGENT_DATA_DICTIONARY.md
│   ├── processed/
│   │   ├── Core Rules.corrections.json  # Versioned visual-text corrections
│   │   ├── Core Rules.json
│   │   ├── Core Rules.md
│   │   ├── Core Rules.chunks.jsonl
│   │   └── images/Core Rules/
│   │       ├── picture-*.png
│   │       └── manifest.json
│   └── indexes/core_rules/
│       ├── chroma/
│       ├── bm25.pkl
│       ├── bm25_chunk_ids.json
│       └── index_manifest.json
├── ingestion/
│   ├── __init__.py
│   ├── pdf_parser.py
│   ├── chunker.py
│   └── build_index.py
├── retrieval/
│   ├── __init__.py
│   └── rulebook_retriever.py
├── sql_retriever/          # Hybrid fixed/generated SQL retrieval and read-only safety
├── llm/                    # Shared Gemini configuration, pacing, retries and usage tracking
├── agent/                 # State, planning, retrieval/answer graphs, turn runner and traces
├── ui/
│   ├── cli.py             # python -m ui.cli
│   └── tui.py             # python -m ui.tui
├── evaluation/
│   ├── datasets/          # Frozen Rulebook, SQL and combined benchmarks
│   ├── adapters/          # Runtime trace to evaluation-record conversion
│   ├── metrics/           # Deterministic retrieval, SQL and Router metrics
│   ├── run.py             # Unified benchmark runner
│   ├── ragas_adapter.py   # Optional RAGAS judge integration
│   ├── runs/              # Generated raw evaluation outputs; kept local
│   └── reports/           # Generated reports; one example run is committed
├── tests/                 # Offline ingestion, retrieval, Agent, UI and evaluation tests
├── docs/
│   ├── eval_protocol.md
│   └── project-guide.md
├── .env.example
├── Makefile
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```



## PDF Processing Design

`ingestion/pdf_parser.py` uses Docling to extract text, headings, tables, page metadata, and pictures from the rulebook.

```text
Core Rules.pdf
      ↓
Docling parser
      ├── Structured JSON
      ├── Markdown preview
      └── PNG images + manifest
```


- JSON preserves the document structure and will be used for chunking.
- Markdown is used only for reading and checking the parsed content.
- Pictures are saved separately as PNG files.
- `manifest.json` connects each picture ID to its file, source, page, and bounding box.
- OCR is optional because the PDF already contains a text layer.
- Parsing does not use an LLM.

The parser also exposes a LangChain-compatible page adapter and reports missing or incomplete conversion outputs clearly.

The processed files are saved in `data/processed/`: `Core Rules.json`, `Core Rules.md`, PNG images, and `images/Core Rules/manifest.json`. The current conversion processed 60 pages and exported 148 images.

### Documented visual-text correction

Our RAG pipeline currently works with text chunks, so PDF images are not sent to Gemini during a query. Sending page images for every request would increase token use and latency, while manually describing every image would also require substantial work.

We therefore tested a smaller approach on one verified visual detail. On page 42, the Grenade Stratagem has a `1CP` label next to its rule text. Docling extracts the label and the rule text, but does not connect them. We converted this visual information into the text `COST: 1CP` and added it to the Grenade chunk before indexing.

The correction is stored in `data/processed/Core Rules.corrections.json` and is applied again whenever chunks are rebuilt. The Grenade test shows that this text representation can be retrieved and used in the final answer.

This does not replace image retrieval. It only shows that important visual details can be added to a text-first RAG pipeline in a simple and reproducible way.

## Chunking Design

Implemented in `ingestion/chunker.py`.

- Input: `data/processed/Core Rules.json`. Markdown is only a human-readable preview.
- Resolve Docling `$ref` values in `body.children` to preserve the original document order.
- Use `section_header` values to build `section_path`. A heading and its body text, lists, and small tables form one logical unit, even when they cross PDF pages.
- Exclude page headers, footers, and empty elements. Pictures are not embedded as text; their IDs, paths, and pages remain attached as metadata.
- Use document-aware recursive splitting only when a logical unit is too long: target 300–400 BGE tokens, maximum 450 tokens, split at paragraph or sentence boundaries, with a 60-token overlap only between parts of the same long unit.
- Output: `data/processed/Core Rules.chunks.jsonl`. The current rulebook produces 212 chunks.
- If `<input stem>.corrections.json` exists, apply and validate those documented visual-to-text corrections before writing the final chunk file.

Rules are kept by section because a heading, its conditions and its effect often need to be read together. The 300–400-token target keeps most rules intact while avoiding overly broad chunks in retrieval and answer prompts. A 450-token limit prevents very long sections from bringing unrelated material into the context. The 60-token overlap is used only when one long section must be split, so boundary information is retained without duplicating the whole corpus.

Each chunk stores:

```text
chunk_id, logical_unit_id, part_index, part_count
text, content_type, section_path, source
pdf_page_numbers, element_refs, source_spans
picture_ids, image_paths, token_count
```

`logical_unit_id` connects parts of one long rule; `chunk_id` remains unique for embedding and retrieval.

## Embedding Design

Implemented in `ingestion/build_index.py`.

The embedding model is the local `BAAI/bge-base-en-v1.5`. It is a general-purpose English retrieval model, which fits the English Rulebook and the expected user questions. It provides good semantic retrieval quality while remaining small enough to run locally on our hardware. Using a local model also avoids an embedding API key, online cost and network dependency when rebuilding the index.

Each embedding contains the chunk text and its section context. The model produces 768-dimensional vectors, which are L2-normalised before being stored in Chroma. A content-hash cache avoids re-embedding chunks that have not changed; changing either the chunk text or the model creates new embeddings.

## Dense and Keyword Index Design

Implemented in `ingestion/build_index.py`.

After embedding the Rulebook chunks, the system builds two indexes from the same `Core Rules.chunks.jsonl` file:

- **Dense index:** ChromaDB `PersistentClient` stores each chunk ID, text, filterable metadata and 768-dimensional embedding in `data/indexes/core_rules/chroma/`.
- **Keyword index:** `rank_bm25` creates `data/indexes/core_rules/bm25.pkl`. The accompanying `bm25_chunk_ids.json` file maps each BM25 result back to its chunk ID.
- **Index manifest:** `index_manifest.json` records the chunk-file hash, chunk count, embedding model, vector dimension and Chroma collection name. This lets the project detect when an index no longer matches the current chunks.

Dense retrieval helps when a user asks about a rule in different words from the Rulebook, while BM25 is useful for exact game terms, unit names, numbers and short rule phrases. The Rulebook Retriever uses both indexes; its fusion and reranking steps are described in the next section.

## Reproducible Rulebook Build

The parsed JSON, readable Markdown, final chunks and correction file are committed to the repository. Extracted PNG images and retrieval indexes are larger rebuildable artifacts, so they remain local.

Running `make prepare` regenerates the images and both indexes from the committed PDF, processing code and correction file. This keeps the submitted Rulebook pipeline reproducible without committing large generated files.

## Rulebook Retriever Design
Implemented in `retrieval/rulebook_retriever.py`.

The Rulebook Retriever only retrieves and ranks Rulebook evidence. The Planner creates the initial `RulebookQuerySet`, which contains the original question, a semantic query and keyword terms. For dependent combined questions, the Agent may refine this query after SQL retrieval. The Retriever itself does not choose routes, call Gemini or generate answers.

```text
RulebookQuerySet
│
├── original_query
│     The user's original question. Kept unchanged for exact terminology,
│     unit names, numbers, dice notation and source display.
│
├── semantic_query
│     A complete, self-contained English rulebook question generated by the Planner.
│     Used for semantic vector retrieval.
│
└── keyword_query
      A list of 3–8 concise English rulebook terms or phrases generated by
      the Planner. Used for keyword retrieval.
      │
      └── Fallback when semantic_query or keyword_query is missing:
            semantic_query = original_query
            keyword_query = [original_query]


RulebookRetriever
│
├── On the first Rulebook request, load persistent indexes and chunk metadata
│     ├── Chroma PersistentClient
│     │     └── data/indexes/core_rules/chroma/
│     │           └── persistent HNSW vector index using cosine similarity
│     ├── rank_bm25 index
│     │     └── data/indexes/core_rules/bm25.pkl
│     ├── BM25 position → chunk ID mapping
│     │     └── data/indexes/core_rules/bm25_chunk_ids.json
│     └── full chunk metadata mapping
│           └── data/processed/Core Rules.chunks.jsonl
│                 └── chunk_id → full chunk record
│
├── Dense Retrieval A: semantic rewrite
│     ├── lazy-load BAAI/bge-base-en-v1.5 on its first Dense retrieval
│     ├── reuse the loaded embedding model for later requests in the same process
│     ├── semantic_query
│     ├── prepend the BGE query instruction
│     ├── generate a normalised 768-dimensional query embedding
│     ├── send the vector to Chroma
│     ├── Chroma searches its persistent HNSW cosine index
│     └── return the default top 12 semantically similar chunk IDs
│
├── Dense Retrieval B: unchanged user question
│     ├── original_query
│     ├── use the same BGE instruction, embedding model and Chroma index
│     └── return the default top 12 chunks without depending on the Planner rewrite
│
├── Inner Dense Fusion
│     ├── apply RRF to the semantic-query and original-query Dense rankings
│     ├── deduplicate by chunk_id
│     ├── convert the fused result into one ordinal Dense ranking
│     └── keep the default top 12 Dense candidates
│
├── BM25 Original-Query Retrieval
│     ├── original_query
│     ├── apply the same lowercase regex tokenisation used at indexing
│     ├── score the chunk corpus with rank_bm25
│     └── return the default top 12 exact-term chunk IDs
│
├── BM25 Keyword Retrieval
│     ├── join keyword_query into one keyword search string
│     ├── apply the same BM25 tokenisation
│     ├── score the chunk corpus with rank_bm25
│     └── return the default top 12 canonical-rule-term chunk IDs
│
├── Candidate Collection
│     ├── combine the fused Dense ranking and the two BM25 rankings
│     ├── deduplicate by chunk_id
│     ├── retain fused dense_rank, bm25_original_rank, and bm25_keyword_rank
│     └── use chunk_id to retrieve full source metadata when needed
│
├── Reciprocal Rank Fusion
│     │
│     │   RRF_score(chunk) =
│     │       1 / (60 + dense_rank)
│     │     + 1 / (60 + bm25_original_rank)
│     │     + 1 / (60 + bm25_keyword_rank)
│     │
│     ├── a missing route rank contributes no score
│     ├── chunks returned by multiple routes receive a higher score
│     ├── sort by RRF score, then best route rank, then chunk ID
│     └── keep the default top 12 fused candidates
│
├── Reranking
│     ├── lazy-load local cross-encoder/ms-marco-MiniLM-L6-v2 on the first reranking request
│     ├── reuse the loaded reranker for later requests
│     ├── score up to 12 pairs: (semantic_query, candidate_chunk.text)
│     ├── preserve one candidate only when at least two of the three fused routes rank it first
│     ├── use reranker scores to order the remaining candidates
│     └── keep the default final top 5 source chunks
│
└── Return RetrievedRulebookContext
      ├── original_query, semantic_query, keyword_query
      └── final ranked chunks with:
            chunk_id, text, section_path, source, pdf_page_numbers,
            logical_unit_id, part_index, part_count, element_refs,
            source_spans, picture_ids, image_paths, route ranks,
            RRF score, and reranker score

Final top 5 source chunks
│
└── passed to the answer-generation layer as Rulebook evidence.
    RulebookRetriever returns evidence only; it does not generate the final answer.
```

The Retriever makes two Dense searches and two BM25 searches. The Dense searches use the same Chroma index: one uses the Planner's semantic query and the other keeps the user's original wording. Their results are merged first into one Dense ranking, so the outer RRF does not treat the same vector index as two separate votes against BM25.

We kept the original-question Dense search after manual checks showed that a Planner rewrite can narrow a broad question. For example, a question about different ways to score points was sometimes rewritten into a query focused only on objective-marker scoring. Keeping the original question gives the Retriever another chance to find the wider set of relevant rules. BM25 is also run twice, but the two routes stay separate: the original-question search preserves exact terms such as unit names, numbers and dice notation, while the keyword search adds canonical Rulebook terms selected by the Planner.

The outer RRF combines the fused Dense ranking with the two BM25 rankings. It uses rank positions rather than raw scores because Chroma similarity and BM25 scores use different scales. If a candidate is ranked first by at least two of these three routes, it is treated as a consensus candidate and keeps one place in the final result. This protects strong agreement between retrieval routes if the reranker scores that chunk lower. Only one place is reserved in this way, so the reranker still decides most of the final context.

The reranker uses the semantic query because it is a complete English question. This is better than reranking with a short follow-up such as “What about it?”, where the subject may be missing.

A semantic rewrite can sometimes make a broad question too narrow. We therefore use it only for reranking, while the earlier Dense and BM25 searches still use the original user question. In this way, the reranker receives a clear question, but the system still keeps the user's original wording as a safeguard. The Planner is also instructed not to remove important parts of the question.

The reranker is `cross-encoder/ms-marco-MiniLM-L6-v2`, a lightweight English model trained to score the relevance of a passage to a query. It can rerank 12 candidates locally on our hardware, without another hosted model or API call. The final top 5 chunks are sent to Gemini as evidence. This keeps the answer context focused while limiting local reranking cost, prompt size and latency.

## SQL Retriever Design

`retrieve_sql(question: str)` retrieves facts from SQLite. It combines fixed parameterized queries with LLM-generated SQL and returns structured evidence for the answer layer.

### Routing (`sql_retriever/hybrid.py`)

The LLM selects one route for the whole question:

- **fixed**: one registered function can fully answer the question. Query builders are defined in `sql_retriever/functions.py`.
- **generated**: the question needs another database query, such as a comparison, ranking, or complex filter.
- **unsupported**: the required facts or calculations are outside the database's scope.

```text
Question
  -> LLM routing
       |
       +-- unsupported -> Return limitation
       |
       +-- fixed / generated
             -> Resolve entities
                  |
                  +-- Unresolved or ambiguous -> Request clarification
                  |
                  +-- Resolved
                        +-- fixed -> Build parameterized SQL
                        +-- generated -> Generate SQL and check entity constraints
                              |
                              -> Read-only execution
                              -> Structured result
                              -> Downstream answer generation
```

Fixed requests normally use one model call. Generated requests use two, or three if SQL repair is needed. 
Answer generation is a separate call.

Both routing and SQL generation receive the complete data dictionary. 
It describes available tables, columns, and relationships.

Separate generation rules define how to query the data. These rules take precedence over examples in the dictionary.

### Entity Recognition and Resolution (`sql_retriever/entities.py`)

The LLM extracts entity mentions and their context during routing. The application looks them up in SQLite to obtain verified identities.

Supported entities include units, factions, sources, abilities, detachments, enhancements, and stratagems.

For “Show abilities for Warboss”, the routing output can contain:

```json
{
  "mode": "fixed",
  "function": "get_unit_abilities",
  "arguments": {"d_id": "@unit"},
  "entities": [
    {
      "key": "unit",
      "kind": "unit",
      "match": "name",
      "value": "Warboss"
    }
  ],
  "message": ""
}
```

`@unit` refers to the entity declaration. The application replaces it with the verified database ID before building SQL.

Faction, source, and owning-unit context narrow the lookup. Explicitly supplied IDs also undergo database lookup and retain leading zeros.

### Ambiguity and Spelling Errors (`sql_retriever/entities.py`)

Names use Unicode, case, and whitespace normalization, followed by:

1. Exact matching.
2. Complete-word phrase matching.
3. Spelling suggestions using Python's `difflib`.

A unique exact or phrase match resolves automatically. Spelling suggestions always require confirmation.

Missing matches, multiple matches, and incomplete candidate sets return `needs_clarification`. Candidates include context such as faction, source, or owning unit.

Generation can also request clarification for undefined criteria such as “best” or an unspecified points configuration (`sql_retriever/hybrid.py`).

### Query Semantics (`sql_retriever/hybrid.py`, `sql_retriever/functions.py`)

Generation rules and fixed query builders account for the database's domain conventions:

- Unit-card ability names and descriptions override catalog values.
- Points belong to a specific unit configuration; results include that configuration.
- Counts use distinct units where joins can duplicate rows.
- Statistics stored as text use guarded conversion for numeric operations. Symbolic values remain unchanged when returned.

For example, a value such as `D6` remains a string and is excluded from integer calculations.

### Entity Constraints in Generated SQL (`sql_retriever/constraints.py`)

Generated SQL must preserve verified entity values in named parameters and use them in effective identity filters.

```sql
SELECT d.name
FROM datasheets d
WHERE d.Id = :unit_id
```

Some entities require compound identities:

- Catalog abilities use ability ID and faction ID.
- Unit-card abilities use unit ID and card line.

SQLGlot checks identity filters through the SQL syntax tree, including supported joins, subqueries, and set operations. This check covers entity constraints; general query semantics still depend on the generated SQL.

### Safe Execution (`sql_retriever/safety.py`)

Fixed queries, generated queries, and entity lookups share one bounded SQLite executor.

It enforces:

- Read-only access.
- Allowed tables and functions.
- One SQL statement per execution.
- Parameter validation.
- Execution-time, computation, and result-size limits.

Results that exceed the output budget are marked as truncated.

### Errors and Repair (`sql_retriever/hybrid.py`, `sql_retriever/safety.py`)

Only generated SQL with syntax or name-resolution errors receives one repair attempt. 
The model receives the failed SQL, parameters, and bounded diagnostics. Repaired SQL passes through the same checks.

Empty results, identity-policy failures, permission errors, parameter errors, and execution-budget failures return directly.

A failed fixed query does not automatically switch to generated SQL.

### Results and Answer Generation (`sql_retriever/result.py`, `agent/response.py`)

The retriever returns one of five statuses:

- `ok`
- `empty`
- `needs_clarification`
- `unsupported`
- `error`

Results include applicable rows, columns, SQL, parameters, strategy, clarification candidates, and truncation or error information.

The downstream answer layer uses this evidence to generate an answer with source references. It preserves clarification choices and explains missing or truncated evidence. An empty result means the query found no matching evidence; truncated results support only partial conclusions.

## Agent Graphs and Conversation Replies

The agent uses LangGraph to plan each request, collect evidence, generate a reply, and update conversation state.

### Workflow (`agent/graph.py`)

`build_agent_graph()` runs the complete turn. `build_evidence_graph()` stops after evidence collection for inspection and evaluation.

```text
User input + conversation context
  -> prepare_turn
  -> plan
       +-- sql: SQL retrieval
       +-- rulebook: Rulebook retrieval
       +-- both: SQL retrieval -> optional query refinement -> Rulebook retrieval
       +-- direct / clarify / unsupported: Skip retrieval

Selected path
  -> finish_evidence
  -> respond
  -> finalize
```

`prepare_turn` clears previous working results while retaining conversation context and pending clarification (`agent/nodes.py`).

### Planning and Source Routing (`agent/nodes.py`)

One structured model call produces a standalone question and selects a route:

| Route | Purpose |
|---|---|
| `sql` | Stored unit facts, attributes, and relationships. |
| `rulebook` | Core Rules, terms, and procedures. |
| `both` | Questions requiring both sources. |
| `direct` | Greetings and application help. |
| `clarify` | Unresolved conversational references or missing conditions. |
| `unsupported` | Requests outside the available sources and capabilities. |

For `sql`, the standalone question passes directly to SQL retrieval. For `both`, the planner extracts a natural-language database subquestion. Rulebook requests receive an English semantic query and keyword terms.

The SQL retriever owns schema interpretation, entity lookup, and SQL generation. Pydantic validates the plan and route/query consistency.

### Conversation References and Clarification (`agent/nodes.py`, `agent/response.py`)

Planning uses recent turns, explicit conversation conditions, and pending clarification candidates to resolve references.

```text
Previous question: Which abilities does Warboss have?
Current question: What about its Toughness?
Standalone question: What is the Toughness of Warboss?
```

Names, exact IDs, and user conditions are preserved. Current explicit conditions override earlier ones. A new topic clears previous conditions and pending clarification.

For a reply such as “the second one”, the planner uses the stored candidate order to reconstruct the pending request. Clarification replies append the exact numbered candidates supplied by retrieval.

Entity ambiguity and spelling suggestions remain the SQL retriever's responsibility.

### Combining SQL and Rulebook Retrieval (`agent/graph.py`)

The `both` route runs SQL first and supports two cases:

- **Independent retrieval**: the rulebook query is already complete. For example, “Get Warboss Toughness and explain Advance.”
- **Dependent retrieval**: SQL facts determine the rulebook query. For example, “Get Warboss abilities and explain their referenced Core Rules.”

Dependent retrieval uses at most one model call to refine the rulebook query from returned SQL evidence.

SQL clarification stops both cases. SQL empty, error, or unsupported results stop dependent retrieval. Independent rulebook retrieval can continue and provide partial evidence.

### Evidence Status and Context Limits (`agent/graph.py`, `agent/response.py`)

The application combines source results into an evidence status. It distinguishes complete evidence, partial evidence, empty searches, clarification, unsupported requests, and failures.

`ok` means the requested sources returned data. It does not measure semantic correctness.

Model context is bounded:

- SQL: at most 50 rows within 32 KiB.
- Rulebook answers: at most five complete chunks within 32 KiB.

SQL refinement uses the same SQL context limits. Whole rows or chunks are removed when necessary. Original retrieval results remain available, and context clipping is recorded separately from retrieval truncation.

### Answer Generation and Citations (`agent/response.py`)

`respond` makes one structured model call using the current question, supplied evidence, source statuses, and limitations. It generates the reply in the user's language.

Factual answers use this turn's evidence. Rulebook chunks receive references such as `[R1]`; the SQL result receives `[S1]`.

The application validates citation IDs and maps them to original chunks or this turn's SQL result. Citation validation checks reference validity; semantic support still requires evaluation.

The application computes the final outcome. Partial evidence, clipped answer context, reported insufficient evidence, or missing citations for required source types prevent a `completed` outcome.

The outer workflow normally uses two model calls: planning and response. Dependent `both` adds one refinement call. SQL retriever calls are additional.

### Conversation Memory (`agent/state.py`, `agent/response.py`)

Memory stores conversational scope: entity names and IDs, topic, and filtering conditions.

The response call can propose up to eight updates. User-derived values must appear verbatim in the current input. SQL-derived identities must match unique values from recognized columns in an untruncated result.

Statistics, rule text, and previous answers are excluded from scope memory. Each value records its source turn. Changing an entity name or ID clears an unmatched identity partner.

Memory updates use the response call; there is no separate memory-extraction call.

### Sessions and Runtime Dependencies (`agent/graph.py`, `agent/nodes.py`)

Reusing the graph and `thread_id` continues a session. A new `thread_id` isolates conversation state.

`InMemorySaver` stores checkpoints within the process. Committed history is limited to 8 full turns within an approximate 5,000-token budget.

`EvidenceContext` carries clients, model configuration, retrievers, and callbacks outside checkpointed state. Pass it to every graph invocation. Reusing it also reuses the lazily loaded rulebook retriever.

Clients and retrievers can be injected for testing.

### Commits and Failure Handling (`agent/response.py`, `agent/runtime.py`)

`finalize` commits a complete user/assistant pair and validated memory updates together. Stable message IDs prevent duplicate messages when the same commit is repeated.

Completed, partial, clarification, and unsupported replies can enter committed history. Failed or cancelled drafts do not.

Direct graph calls propagate model failures and validation errors while retaining the last successful checkpoint. Agent nodes are not retried automatically.

`run_turn` converts exceptions and cancellation into terminal results and records trace events. Cancellation may wait for an in-flight synchronous model call to return.


## CLI and Local Graph Traces

Configure `GOOGLE_API_KEY` in `.env`, provide `data/sql/wahadb.sqlite`, and build the rulebook indexes using the commands below. Run from the project root:

```bash
uv run python -m ui.cli --trace
uv run python -m ui.cli "Which abilities does Warboss have?" --trace --sources
uv run python -m ui.cli --inspect data/traces/SESSION/RUN.jsonl --sources
uv run python -m ui.cli --inspect data/traces/SESSION/RUN.jsonl --json
```

- Omit the question for an interactive session. `/new` starts isolated memory and deletes the old thread's in-memory checkpoints; it keeps its trace files. `/sources` displays all cited sources, `/sources S1` one source, `/trace` toggles live step display, and `/quit` or EOF exits. Ctrl-C at the prompt exits; during a turn it records cancellation and returns to the prompt. Single-question error/cancellation exits with status 1/130; other outcomes return 0.
- Answers and requested source JSON go to stdout; progress, route, source status/count, session/run/turn IDs, SQL strategy, outcome, total latency and trace path go to stderr. Steps and route reasons are shown with `--trace`. Replies currently appear as whole messages, without token streaming.
- Disk tracing is on even when live step display is off. `--no-trace` disables files and can be combined with `--trace` for display only. `--trace-dir PATH` selects another local directory. No transcript is loaded into a new process automatically.
- `agent.runtime.run_turn(graph, context, *, session_id, question, trace_dir=..., on_event=None, cancel_event=None)` is shared by the CLI and TUI. It consumes LangGraph's native `tasks` and `custom` streams, issues fresh run IDs, and sends the same redacted event records to the callback and JSONL writer. Keep a graph/context and session ID for follow-ups; use caller-generated internal session IDs, not arbitrary paths. `cancel_event` accepts a `threading.Event` for cooperative cancellation between graph events and SDK calls.
- Each run writes `data/traces/<session_id>/<run_id>.jsonl`, with a `session.json` manifest containing creation time, model, Python/LangGraph versions, lock-file hash and working-source hash. The code hash identifies changes but does not retain a runnable code snapshot; data/index fingerprints and historical environments are not captured yet. Scope is graph events plus observed LLM SDK calls/usage; full model requests and retriever internals remain unavailable. Existing session manifests describe their creation-time capabilities; each run's events and summary describe its actual usage coverage.
- Events contain schema version, session/run IDs, sequence, UTC timestamp, event/node, span and parent IDs. They cover `run.started`, node `step.started/finished/failed`, `route.selected`, `query.refined`, `retrieval.completed`, LLM start/wait/retry/finish/failure events, and exactly one `run.finished` for a controlled turn. Node starts snapshot their actual State input; finishes snapshot updates, including history trimming/overwrite, planned queries, bounded retriever results, PDF metadata, SQL provenance, answer, citations and memory commits. These are **graph-node** inputs, not a recording of the exact model HTTP requests.
- Durations use a monotonic clock and measure observed node lifecycles, including stream/trace/display overhead; run time includes cold model initialization, throttle waits and retry backoff. Each LLM attempt records its observed SDK-call duration, including retries internal to that SDK invocation; application-level retry attempts have separate call IDs. SQL's non-LLM internal phases, raw model outputs, repair SQL before/after snapshots and rulebook intermediate candidate rankings remain unavailable.
- The writer flushes each complete JSON line. It does not promise power-loss durability. Write/open/close failures emit a visible `trace.failed` warning and mark the returned trace incomplete while preserving a valid answer. Error records retain exception type, validation locations and stack file/function/line locations, omitting raw provider messages, locals and source lines. Known configured API keys and credential fields are redacted; SDK clients and authentication headers are never serialized from runtime context.
- `--inspect` reads saved records without building/resuming a graph or calling a model/SQL. It shows lifecycle, route and result; `--sources` resolves actual cited chunks or executed SQL, bindings, full bounded rows and separate prompt-clipping flags. `--json` exposes full recorded node snapshots. An incomplete final JSON fragment is skipped with a warning; a run without `run.finished` is `interrupted`. Corrupt middle records or mixed run IDs/sequence are rejected. This is replay of recorded facts, not a rerun of the original experiment.

The lower-level graph still propagates exceptions. The runner's returned outcome is authoritative for failed/cancelled turns: its returned State copy clears answer/citations; the underlying checkpoint retains the last successful node for debugging. Cancellation does not forcibly terminate synchronous provider/model calls; it may wait for an in-flight call to return before the stream stops. No new turn is submitted during that wait. Once `finalize` starts, the memory commit completes and the committed outcome wins over a late cancellation. Automatic checkpoint rotation for long sessions and detailed retriever instrumentation remain pending.

### Textual TUI

```bash
uv run python -m ui.tui
uv run python -m ui.tui --model gemini-3.5-flash-lite
uv run python -m ui.tui --inspect data/traces/SESSION/RUN.jsonl
```

`ui/tui.py` uses a Textual thread worker to call the same runner as the CLI. Only redacted trace messages cross into the UI through thread-safe `post_message`; stdout/stderr output is not parsed. One turn runs at a time. Input, New and the turn selector are disabled until the worker returns. Esc requests cancellation; Ctrl+Q requests cancellation and quits after the worker returns.

The chat is paired with an inspector showing session/run/turn/model, route and `both` dependency, observed step/total timings, SQL strategy/row count, rulebook count, clipping indicators, trace status and failed node. LLM child rows show each actual SDK call; metadata shows per-turn and session calls/tokens/coverage. Select a turn to load its retained JSONL on demand; Enter on a step opens its complete redacted input/output in Payload. Select `run.finished` for the complete final state, usage breakdown, error/stack locations, context and clarification metadata. Sources resolves cited SQL/parameters/rows and rulebook chunk IDs/pages/text/scores through `source_details`. The inspector can be toggled with Ctrl+T or `/trace`; below 100 columns it replaces the chat view. Ctrl+S or `/sources [ID]` opens sources.

`/new` starts a fresh session, deletes the old in-process checkpoints and clears the UI's run list, while retaining disk traces. The chat rendering buffer is limited to 1,000 lines; only the selected run's full events/state are retained in UI memory. `--no-trace` keeps current inspection but disables access to earlier turn details. `--inspect` is read only: it creates no graph/client and executes no model or SQL calls. It also displays interrupted traces and reader warnings. Live replies appear as complete answers; no answer-token streaming or retriever-internal timings are implied.

Headless UI tests use Textual's `run_test`/Pilot with `unittest.IsolatedAsyncioTestCase`, real graph orchestration and offline model/retriever stubs; no real API calls are required. See [Textual workers](https://textual.textualize.io/guide/workers/) and [testing](https://textual.textualize.io/guide/testing/).

### LLM calls and provider token usage

`llm/instrumentation.py` is the shared boundary for application Gemini calls. Plan, dependent rulebook-query refinement, response generation, SQL routing/generation/repair and the legacy fixed-function entry all use it. The boundary applies a process-wide minimum interval between real SDK calls so independent graph nodes cannot collectively exceed the free-tier rate too quickly. This deliberately trades evaluation and chat throughput for fewer provider-side failures; injected offline clients are not delayed.

The default interval is five seconds. Temporary `429`, `500`, `502`, `503`, `504`, and SDK `ServerError` failures use bounded exponential retries, with three total attempts and a five-second base delay by default. `GEMINI_MIN_INTERVAL_SECONDS`, `GEMINI_MAX_ATTEMPTS`, and `GEMINI_RETRY_BASE_SECONDS` override these values; an interval of zero disables pacing. Permanent client, authentication, model, validation and policy failures are not blindly retried. Throttling reduces free-tier rate pressure but cannot guarantee availability when Google is overloaded or a daily quota is exhausted.

Each actual SDK attempt emits `llm.started` and one `llm.finished`/`llm.failed`, with a unique call ID parented to its graph node. Rate-limit waits emit `llm.waiting`, and a recoverable provider failure emits `llm.retrying`; the CLI and TUI display both instead of appearing frozen. Events retain purpose, attempt number, requested model, provider, monotonic duration, response ID/model version when supplied, and usage. Provider error text, credentials, prompts and raw responses are not copied into LLM metadata events. Usage is captured before JSON/citation validation, so invalid structured output still counts. `llm_calls` counts SDK invocations including failed retry attempts; it does not claim to count HTTP retries hidden inside one SDK invocation or evaluation-judge calls.

Provider usage preserves nullable `prompt_token_count`, `candidates_token_count`, `thoughts_token_count`, `cached_content_token_count`, `tool_use_prompt_token_count` and `total_token_count`. Values are reported, never estimated; cached tokens are not added to prompt tokens and the provider's total is not reconstructed from a formula. API errors or an absent usage response have `usage: null` at the call level. A missing field stays null; an explicitly reported zero stays zero.

`run.finished` and the runner result expose `llm_calls` and a `usage` summary. Each token total is null unless every observed call reports that field and the instrumented scope is complete. `known_totals` holds only reported subtotals, `reported_calls` gives coverage per field, and `total_calls` is the observed-call denominator. `complete` requires complete prompt/candidate/total coverage; optional thinking/cache/tool fields may remain null. A question-only injected SQL callable is opaque when executed: `instrumentation_complete=false` marks that boundary, while retaining counts and known usage from instrumented calls. There are no usage counters in checkpointed `AgentState` or future prompts.

CLI/TUI also reduce run summaries into session totals, keyed by run ID; history selection never adds the same run again. `/new` resets totals. `--no-trace` still records live statistics in memory. Old traces stay unavailable and are not estimated from node counts. Evaluation's trace adapter passes through the same fields. During cancellation the runner waits for in-flight work and drains retained LLM events before writing `run.finished`, preserving any returned usage even when no answer is committed. A killed process can leave only `llm.started`; its consumed tokens remain unknown.

## Build and Simple Verification

Install `uv` and GNU Make 4.3 or newer, then configure `GOOGLE_API_KEY` in `.env`
(copy `.env.example`) or in the environment. From the repository root:

```bash
make prepare
```

The copied `.env.example` also contains the free-tier pacing defaults. They can
be increased for stricter quotas or reduced for a paid project; keeping the
defaults are recommended for the course evaluation run.

On macOS, use `gmake` if GNU Make is installed under that name. Preparation uses
uv-managed Python (including headers for native builds) and syncs the locked
runtime and `eval` dependencies. A C++ compiler is needed when dependencies such
as `scikit-network` build from source. Preparation checks the local key configuration,
reads the data dictionary, queries SQLite in read-only mode, and builds the
rulebook artifacts. It then validates the existing benchmark datasets and runs
one local retrieval request, downloading the reranker on first use. It makes no
Gemini requests and does not run evaluation or generate benchmark questions.
The key's validity, model access, and quota are exercised by later model calls.

The committed inputs are `data/pdf/Core Rules.pdf`, `data/sql/wahadb.sqlite`,
`data/sql/SQL_AGENT_DATA_DICTIONARY.md`, the source-backed
`data/processed/Core Rules.corrections.json`, and `evaluation/datasets/*_v1.jsonl`.
Parsed JSON/Markdown, images, chunks, and indexes are local generated artifacts.

`make parse`, `make chunk`, and `make index` also build any required earlier
stages. `make check` builds the indexes if needed and runs the same readiness
checks as `make prepare`. Every invocation syncs dependencies; this does not
force unchanged data to rebuild. Make tracks the PDF, processing code,
corrections, dependency files, and key output files. A correction change rebuilds
chunks and indexes; an indexing-code change only rebuilds indexes. Changes to
the parser, PDF, Makefile, or dependency files rebuild the full chain. Index
rebuilds replace the Chroma collection and reuse cached embeddings for unchanged
content. Failed recipes remove their modified targets so a later run can retry.
Use `make -B parse` to force parsing, for example after manually deleting an
individual picture crop.

After preparation, start the CLI/TUI as described above, or run evaluation:

```bash
uv run --env-file .env python -m evaluation.run --dataset all
uv run --group eval --env-file .env python -m evaluation.run --dataset all --ragas
```

The first command runs deterministic evaluation and the real Agent with the
application throttle. Run the second command only after that succeeds: it adds
RAGAS judge calls through a separate client and may require additional provider
quota. See `docs/eval_protocol.md` for isolated dataset commands and report
interpretation.

For individual processing commands and the unit tests:

```bash
uv sync --locked --group eval --managed-python
uv run python -m ingestion.pdf_parser "data/pdf/Core Rules.pdf"
uv run python -m ingestion.chunker "data/processed/Core Rules.json"
uv run python -m ingestion.build_index "data/processed/Core Rules.chunks.jsonl" --recreate
uv run python -m retrieval.rulebook_retriever "How does a wound roll work?" \
  --semantic-query "How is a wound roll resolved in the Core Rules?" \
  --keyword-query "Wound Roll" Strength Toughness
uv run python -m unittest discover -s tests -v
```

Add `--ocr` to the parser command when OCR is needed. The first run may download Docling model files.
The first chunking and indexing run also downloads `BAAI/bge-base-en-v1.5`.

## Evaluation

The project uses frozen Rulebook, Text-to-SQL and combined benchmarks.

- **Rulebook:** checks whether the Retriever returns the Rulebook chunks needed to answer a question. Each run computes deterministic retrieval metrics, and RAGAS can be enabled to assess final answers when Gemini is available.
- **Text-to-SQL:** compares the returned rows and status with the reviewed expected result from the project SQLite database.
- **Combined questions:** check that the Agent uses both sources and returns evidence from both of them.
- **Runtime traces:** record the selected route, retrieved evidence, final answer, latency, LLM calls and token counts when the provider reports them.

Each run saves raw results, failed cases and a Markdown report. The full benchmark design, datasets, commands and report format are described in the [Evaluation Protocol](eval_protocol.md).

An [example evaluation report](../evaluation/reports/20261028_173548/report.md) is included as a reference output. It records one complete run, including its configuration and failures; it is not presented as a fixed pass threshold.

## Final Submission Checklist

- confirm that the two member names shown in `README.md` use the preferred spelling;
- run the complete offline test suite;
- run the deterministic Rulebook, Text-to-SQL, and combined benchmarks from a
  committed working tree, then run RAGAS separately if quota permits;
- manually inspect the two combined answers and the multi-turn acceptance cases;
- retain one reviewed evaluation report, including failures and configuration,
  as the final project result;
- verify the setup and Quick Start with a fresh `GOOGLE_API_KEY`, then send the
  repository link to the teacher.
