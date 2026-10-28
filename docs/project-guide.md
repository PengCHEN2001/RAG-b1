# Project Guide

## Overview

RAG-b1 is a Warhammer 40K question-answering project using a core rulebook PDF and SQLite unit data through a LangGraph agent, with CLI / Textual TUI interfaces and local graph traces.

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
│       ├── Core Rules.corrections.json  # Versioned input; other files are generated
│       ├── Core Rules.json
│       ├── Core Rules.md
│       ├── Core Rules.chunks.jsonl
│       └── images/Core Rules/
│           ├── picture-*.png
│           └── manifest.json
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
├── agent/                 # State, planning, retrieval/answer graphs, turn runner and traces
├── ui/
│   ├── cli.py             # python -m ui.cli
│   └── tui.py             # python -m ui.tui
├── tests/
│   ├── test_pdf_parser.py
│   ├── test_chunker.py
│   └── test_rulebook_retriever.py
├── docs/
│   ├── journal.md
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

The parser also provides a LangChain page adapter. Missing files and incomplete conversions produce clear errors.

The processed files are saved in `data/processed/`: `Core Rules.json`, `Core Rules.md`, PNG images, and `images/Core Rules/manifest.json`. The current conversion processed 60 pages and exported 148 images.

### Documented visual-text correction

The project currently follows a text-first RAG strategy. Extracted images and their page references are retained for provenance, but full PDF page images are not sent to Gemini at query time. This avoids the additional image-input tokens, latency, and implementation complexity of multimodal generation.

For a verified visual element that directly affects the meaning of a rule, the project uses a scoped visual-to-text normalisation. On page 42, the Grenade Stratagem displays its `1CP` cost in a graphical label beside the rule block. Docling extracted the main rule text but did not preserve the spatial relation between that label and the Grenade section, so the production chunk represents the visible label as `COST: 1CP` before indexing.

This source-backed correction is recorded in `data/processed/Core Rules.corrections.json`. The chunker automatically loads a correction file with the same stem as its input JSON, validates the target chunk and PDF page, and then applies the documented text insertion. The correction restores information visible on PDF page 42; it does not add a rule that is absent from the source or modify evaluation references only.

The benchmark verifies that this searchable text representation supports the expected answer. This demonstrates the feasibility of selective visual-to-text normalisation, but it is not presented as a general image-understanding solution. A future layout-aware extractor could associate all Stratagem cost icons automatically.

## Chunking Design

Implemented in `ingestion/chunker.py`.

- Input: `data/processed/Core Rules.json`. Markdown is only a human-readable preview.
- Resolve Docling `$ref` values in `body.children` to preserve the original document order.
- Use `section_header` values to build `section_path`. A heading and its body text, lists, and small tables form one logical unit, even when they cross PDF pages.
- Exclude page headers, footers, and empty elements. Pictures are not embedded as text; their IDs, paths, and pages remain attached as metadata.
- Use document-aware recursive splitting only when a logical unit is too long: target 300–400 BGE tokens, maximum 450 tokens, split at paragraph or sentence boundaries, with a 60-token overlap only between parts of the same long unit.
- Output: `data/processed/Core Rules.chunks.jsonl`. The current rulebook produces 212 chunks.
- If `<input stem>.corrections.json` exists, apply and validate those documented visual-to-text corrections before writing the final chunk file.

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

- Model: local `BAAI/bge-base-en-v1.5`.
- Rationale: English rulebook, no API key or online cost, and 768-dimensional vectors suitable for the project.
- Input: each chunk's section context and text.
- Vectors are L2-normalised. A content hash cache avoids re-embedding unchanged chunks; changing either the chunk text or model creates new embeddings.

## Dense and Keyword Index Design

Implemented in `ingestion/build_index.py` from the same `Core Rules.chunks.jsonl` file.

- Dense index: ChromaDB `PersistentClient` stores chunk ID, text, filterable metadata, and its 768-dimensional vector in `data/indexes/core_rules/chroma/`.
- Keyword index: `rank_bm25` creates `data/indexes/core_rules/bm25.pkl`; `bm25_chunk_ids.json` maps each BM25 position back to the same chunk ID.
- `index_manifest.json` records the chunk file hash, chunk count, embedding model, vector dimension, and Chroma collection name.
- The two indexes will later be queried together and combined with Reciprocal Rank Fusion (RRF) for hybrid retrieval. BM25 is stored beside Chroma, not inside it.

All generated data in `data/processed/` and `data/indexes/` is ignored by Git and can be rebuilt from the source PDF.

## Rulebook Retriever Design

Implemented in `retrieval/rulebook_retriever.py`. The Retriever does not route, rewrite, or answer questions. A future Router supplies the query set; the Retriever retrieves and ranks rulebook evidence only.

```text
RulebookQuerySet
│
├── original_query
│     The user's original question. Kept unchanged for exact terminology,
│     unit names, numbers, dice notation, source display, and final answer generation.
│
├── semantic_query
│     A complete, self-contained English rulebook question generated by the Router.
│     Used for semantic vector retrieval.
│
└── keyword_query
      A list of 3–8 concise English rulebook terms or phrases generated by
      the Router. Used for keyword retrieval.
      │
      └── Fallback when Router output is unavailable or invalid:
            semantic_query = original_query
            keyword_query = [original_query]


RulebookRetriever
│
├── Load retrieval resources once at application startup
│     ├── BAAI/bge-base-en-v1.5 embedding model
│     ├── Chroma PersistentClient
│     │     └── data/indexes/core_rules/chroma/
│     │           └── persistent HNSW vector index using cosine similarity
│     ├── rank_bm25 index
│     │     └── data/indexes/core_rules/bm25.pkl
│     ├── BM25 position → chunk ID mapping
│     │     └── data/indexes/core_rules/bm25_chunk_ids.json
│     └── full chunk metadata mapping
│           └── data/processed/Core Rules.chunks.jsonl
│                 └── chunk_id → full ChunkRecord
│
├── Dense Semantic Retrieval
│     ├── semantic_query
│     ├── prepend the BGE query instruction
│     ├── generate a normalised 768-dimensional query embedding
│     ├── send the vector to Chroma
│     ├── Chroma searches its persistent HNSW cosine index
│     └── return top 12 semantically similar chunk IDs
│
├── BM25 Original-Query Retrieval
│     ├── original_query
│     ├── apply the same lowercase regex tokenisation used at indexing
│     ├── score the chunk corpus with rank_bm25
│     └── return top 12 exact-term chunk IDs
│
├── BM25 Keyword Retrieval
│     ├── join keyword_query into one keyword search string
│     ├── apply the same BM25 tokenisation
│     ├── score the chunk corpus with rank_bm25
│     └── return top 12 canonical-rule-term chunk IDs
│
├── Candidate Collection
│     ├── combine the three ranking lists
│     ├── deduplicate by chunk_id
│     ├── retain dense_rank, bm25_original_rank, and bm25_keyword_rank
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
│     └── keep top 12 fused candidates
│
├── Reranking
│     ├── lazy-load local cross-encoder/ms-marco-MiniLM-L6-v2 on the first request
│     ├── reuse the loaded reranker for later requests
│     ├── score 12 pairs: (semantic_query, candidate_chunk.text)
│     ├── preserve one candidate only when at least two retrieval routes rank it first
│     ├── use reranker scores to order the remaining candidates
│     └── keep final top 5 source chunks
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
└── passed to the answer-generation layer with original_query.
    RulebookRetriever returns evidence only; it does not generate the final answer.
```

The current reranker is `cross-encoder/ms-marco-MiniLM-L6-v2`. It is a small English cross-encoder, suitable for reranking only 12 candidates from the English rulebook on a local student laptop. If future evaluation requires stronger semantic ranking or multilingual query support, it can be replaced with, like the larger `Qwen/Qwen3-Reranker-0.6B`; that model requires substantially more local storage and memory.

## SQL Retriever Design

The main entry point is `retrieve_sql(question: str)` in `sql_retriever/hybrid.py`, also exported as `sql_retriever.retrieve_sql`. It prefers registered parameterized functions when one function can answer the whole question and uses LLM-generated SQL for requests outside their coverage.

```text
question: str
  -> Validate input, load the full data dictionary, and prepare the model client
  -> LLM routing: return a structured intent
  -> Validate the JSON, entity declarations, and routing configuration
       |
       +-- unsupported
       |     -> Return RetrievalResult(status="unsupported")
       |
       +-- fixed / generated
             -> For fixed requests, normalize literal IDs into entity references
             -> resolve_entities(): normalized exact / phrase / spelling lookup
                  |
                  +-- No match, spelling suggestions, multiple matches, or truncated candidates
                  |     -> Return needs_clarification with candidates
                  |
                  +-- Unique matches, or no entities to resolve
                        -> resolved entity dictionary
                             |
                             +-- fixed
                             |     -> Validate and substitute entity references
                             |     -> Registered builder returns SQL + parameters
                             |
                             +-- generated
                                   -> LLM receives the question, resolved entities,
                                      generation rules, and full data dictionary
                                   -> Return SQL + parameters, or stop with
                                      needs_clarification / unsupported
                                   -> Check parameters and resolved identity bindings

Executable SQL + parameters
  -> execute_bounded(): authorized, resource-bounded, read-only execution
  -> Assemble and return RetrievalResult
```

The routing intent contains `mode`, `function`, `arguments`, `entities`, and `message`. For example:

```json
{
  "mode": "fixed",
  "function": "get_unit_abilities",
  "arguments": {"d_id": "@unit"},
  "entities": [
    {"key": "unit", "kind": "unit", "match": "name", "value": "Warboss"}
  ],
  "message": ""
}
```

An entity declaration describes an object to look up. Its `key` is a request-local reference name, `kind` selects a lookup definition from `ENTITY_TABLES`, `match` is the string `"name"` or `"id"`, and `value` must appear in the original question. Optional `faction_key`, `source_key`, and `unit_key` refer to context entity keys in the same request, without an `@` prefix. Abilities also accept an explicitly supplied `line` with `unit_key`. The model extracts these lookup descriptions; the application obtains the actual identities from SQLite.

`validate_entities()` checks the declarations, their occurrence in the question, and context references against the target kind and available foreign keys. On the fixed path, `_normalize_fixed_references()` converts literal `d_id` or `faction_id` arguments into `@entity_key` references, adding and validating ID-based entity declarations where necessary. This sends explicitly supplied IDs through the same database verification as names.

`resolve_entities()` in `sql_retriever/entities.py` resolves factions and sources first, then units before their abilities. Units support both faction and source constraints; abilities additionally support their owning unit. Other faction-linked entities support faction constraints. Explicit context references apply only to their referring entities. A single unreferenced context is shared; multiple unreferenced contexts of one applicable kind require clarification. All applicable filters intersect before name matching, including spelling suggestions. Names use Unicode NFKC, case folding, and whitespace normalization, followed by exact matching, complete-word phrase matching, and `difflib` spelling suggestions. Unique exact or phrase matches resolve automatically; spelling suggestions always require confirmation. IDs use exact string matching and retain leading zeros. It always returns an internal `EntityResolution` with `status`, `resolved`, `candidates`, `message`, and `truncated`. The `resolved` dictionary is indexed by entity key:

```json
{
  "unit": {
    "key": "unit",
    "kind": "unit",
    "id": "000000001",
    "name": "Warboss",
    "faction_id": "ORK",
    "faction_name": "Orks",
    "source_id": "000000046",
    "source_name": "Orks",
    "matched_text": "Warboss",
    "match_type": "exact"
  }
}
```

Uniqueness is determined under the lookup table and applicable filters; an ID is not assumed to be globally unique. Missing, ambiguous, or spelling-only matches return an `EntityResolution` with `status="needs_clarification"`, retaining any identities already resolved. `hybrid.py` converts this into the public `RetrievalResult(status="needs_clarification")` and stops retrieval. Successful resolution uses `status="resolved"`; an empty entity list produces the same status with an empty resolved dictionary and requires no lookup queries.

Candidates include the available faction/source IDs and names, plus `matched_text` and `match_type` to explain the original mention and matching method. Card abilities also include their owning unit and line. Metadata uses left joins so missing optional catalog labels do not remove the entity.

Ability lookup combines the `abilities` catalog with named `datasheets_abilities` rows. Under a unit or source constraint it also includes card rows that inherit their catalog name. A catalog ability keeps its verified `id`; a card ability uses the real `(DatasheetId, line)` identity as `unit_id` and `line`. Its optional `ability_id` is only a catalog link. Card names and descriptions override catalog values, and catalog joins require a matching faction or a shared core ability with an empty faction ID.

Name catalogs are scanned without the business-query row cap, retaining the executor's byte, time, and VM budgets. An incomplete catalog requires narrower context instead of accepting a partial match. Exact and phrase candidates are capped at 200, spelling suggestions at five; omitted candidates set `truncated`. Spelling recall uses `get_close_matches` with a 0.6 cutoff, not a calibrated identity confidence. Catalogs are small enough for a per-lookup scan; no search service or additional dependency is required.

On the fixed path, `_fixed_query()` checks reference types and substitutes verified IDs. Both this path and the legacy JSON selection parser call `build_fixed_query()` to validate model-supplied arguments once before invoking a trusted builder. A role filter must be explicitly supplied in the question. On the generated path, the second model call receives `required_bindings` and returns `mode`, `sql`, `parameters`, and `message`; only `mode="query"` proceeds to execution. The application checks exact parameter values: `<entity_key>_id` for catalog entities, an additional `<entity_key>_faction_id` for catalog abilities (empty for shared core rules), or `<entity_key>_unit_id` and `<entity_key>_line` for card abilities.

`get_unit_model_stats(d_id)` handles one unit's characteristic lookup, including a single value such as Toughness. It returns every model profile with unit/model identity, movement, toughness, save, invulnerable save, wounds, leadership and objective control. Symbolic values remain strings. Cross-unit comparisons, rankings and numeric filters still use generation.

`constraints.validate_entity_constraints()` uses SQLGlot's SQLite AST and scopes to verify effective identity filters in `WHERE` or inner-join `ON` predicates. It accepts direct key equalities and lists of resolved single-key IDs, inherits filters from required input CTEs/derived tables, and checks every set-operation branch. Compound card and catalog-ability identities must constrain the same table alias. Parameter mentions in projections, unused CTEs, scalar subqueries, or optional left-join conditions do not suffice. Unproven forms return `sql_policy`; resolved-entity queries do not accept right/full joins. See `docs/sql_entity_matching.md` for supported forms and limits.

Both model stages receive the complete unchanged `data/sql/SQL_AGENT_DATA_DICTIONARY.md`. Generation rules take precedence over its examples and add semantic guidance, including card-specific ability overrides, distinct counts, cost configurations, and a guarded TEXT-to-integer template. AST validation covers resolved identity filters only; it does not prove general join, numeric, projection, or natural-language semantics. There is no separate live-schema validator.

Fixed queries, generated queries, and entity lookups all use `execute_bounded()` in `sql_retriever/safety.py`. It validates input boundaries, opens a read-only connection, applies business-table and function authorization, executes one statement, and enforces execution and result budgets. It returns `columns`, `rows`, and `truncated`. The outer retriever adds the executed SQL, parameters, strategy, and status: non-empty or truncated results are `ok`; an empty, untruncated result is `empty`. Expected validation, model, and execution failures become `error` results with an `error_code`. Natural-language answer generation belongs to the downstream answer layer.

Only a generated query's syntax or name-resolution failure permits one model repair with the failed SQL, its parameters, and bounded compilation diagnostics. Repair retains the original question and resolved identities and never receives database result rows. Corrected SQL undergoes the same identity and execution checks. Empty results, identity-policy failures, permission denials, parameter errors, and execution-budget failures do not retry. A failed fixed request does not automatically switch to generation. Fixed requests normally use one model call; generated requests use two, or three when a repair is attempted.


## Agent Graphs and Conversation Replies

`agent/nodes.py` provides `prepare_turn(state)` and `plan(state, *, client=None, model=...)`.

- Callers supply the new `user_input` and `run_id`. `prepare_turn` increments the turn index, clears previous working results, and preserves conversation context and pending clarification. It keeps up to eight committed turns within an approximate 5,000-token history budget, using LangGraph's `Overwrite` to replace the message window.
- `plan` makes one structured Google GenAI call to resolve conversational references, choose a source route, and plan source-specific questions. For rulebook routes, it constructs the existing `RulebookQuerySet` and copies the original input directly. The two `both` modes are distinguished by `depends_on_sql`.
- For a pure `sql` route, the application passes `standalone_question` directly to SQL retrieval instead of using a second model-authored query. For `both`, `sql_question` remains the natural-language database subquestion. Planning receives no data dictionary and must preserve supplied names and avoid generating SQL; the SQL retriever owns schema interpretation and receives the dictionary in routing and generation.
- Planning receives history, explicit conversation conditions, and actual pending candidates. The model reports when a new topic should clear old conditions and pending clarification. It does not append an assistant message or run either retriever.
- Clarification replies also receive the actual pending candidates, including their kinds; the application still appends the exact numbered candidate list.
- Inject a GenAI-compatible `client` for testing; the default client is loaded lazily from `llm.llm`. Pydantic validates the response and route/query consistency. Invalid output and provider errors propagate without automatic retries or alternative retrieval.

`agent/graph.py` provides `build_agent_graph()` for complete turns: evidence collection → `respond` → `finalize`. `build_evidence_graph()` retains the evidence-only entry point for retrieval inspection and evaluation.

```python
from uuid import uuid4
from agent import EvidenceContext, build_agent_graph

graph = build_agent_graph()    # Keep this graph for the session.
context = EvidenceContext()    # Default clients/models load only when needed.
config = {"configurable": {"thread_id": "session-1"}}
result = graph.invoke(
    {"user_input": "Which abilities does Warboss have? And find core rules related to them.", "run_id": str(uuid4())},
    config,
    context=context,
)
print(result["answer"], result["outcome"], result["citations"])

# A follow-up uses the same graph, context and thread_id, with a fresh run_id.
followup = graph.invoke(
    {"user_input": "What about its Toughness?", "run_id": str(uuid4())},
    config,
    context=context,
)
print(followup["answer"])
```

- Pass `context` explicitly to every `invoke` / `stream`. Reuse the context to reuse the lazy rulebook instance. Inject `client`, `model`, `rulebook_retriever`, or a `sql_retriever(question)` callable for offline use. Runtime dependencies are excluded from checkpoints.
- Both routes run SQL first. Independent `both` uses the planned rulebook query directly; dependent `both` makes at most one refinement call using returned SQL facts. SQL clarification stops both modes and retains the full request and actual candidates. SQL empty/error/unsupported stops dependent retrieval; independent rulebook retrieval can continue with partial evidence.
- Refinement uses at most 50 SQL rows and 32 KiB of serialized SQL context, removing whole trailing rows. `sql_context_truncated` records this separate from `sql_result.truncated`; the original result is unchanged. With no usable rows, dependent retrieval stops without guessing. SQLite blobs are labeled as hex in the prompt and remain bytes in state.
- `evidence_status` is `ok / empty / partial / needs_clarification / unsupported / error / skipped`; `retrieval_errors` distinguishes rulebook availability failures from empty hits. `ok` means all requested evidence sources returned rows/chunks, not a semantic correctness score. Invalid model output / provider failures propagate and leave the last checkpoint available; the evidence summary remains unset when the graph fails before completion.
- `InMemorySaver` retains node checkpoints under `thread_id`; its serializer explicitly registers the existing result dataclasses and restores tuple fields. Reusing the graph and ID continues a session; a new ID isolates state. It does not persist across process exit, rotate old checkpoints, or provide detailed retriever traces yet.
- To monitor node completion directly, use `graph.stream(..., stream_mode="updates", context=context)`; each update is keyed by node name. For retained traces and node lifecycle timings, use the turn runner below. Nested retriever phases, actual model requests/usage and answer-token streaming remain separate steps.

`agent/response.py` provides `respond(state, *, client=None, model=...)` and `finalize(state)`:

- `respond` makes one structured GenAI call with the original-language question, standalone request, actual evidence, source statuses and limitations. It does not send old answers as factual evidence. The outer workflow uses two model calls normally (plan + respond), three for dependent `both`; SQL's internal calls are additional. Clarification, direct and unsupported replies also use this response call for natural-language wording.
- The application assigns `[R1]` etc. to supplied chunks and `[S1]` to this turn's SQL result. A response may cite one source or combine complementary evidence from several sources; citations can be written separately (`[R1] [R3]`) or as a comma-separated group (`[R1, R3]`). The parser normalises these forms, validates every ID against evidence supplied during the current turn, and stores only cited references: rulebook citations map to original chunk IDs; SQL maps to `<run_id>:sql`, resolvable through `sql_result` including SQL, parameters, rows and truncation. Fact replies require citations unless explicitly reporting insufficient evidence. Unknown or malformed IDs abort without a retry or a committed reply. Reference existence is checked; semantic support and response language still require real-model evaluation.
- Answer SQL context shares the 50-row / 32-KiB limit with refinement. Rulebook context uses at most five complete chunks within 32 KiB of serialized chunk data. `answer_context_truncated` records each source's clipping separately from refinement's `sql_context_truncated` and the original SQL `truncated` flag. Original results remain intact; omitted evidence cannot be cited. These limits do not enforce the planned exact total request-token budget.
- Outcomes are computed by the application. Evidence `empty` becomes `partial`, `skipped` becomes `completed`, and clarification/unsupported/error retain their meanings. Partial retrieval, answer clipping, reported insufficient evidence or failing to cite all requested source types cannot become `completed`. SQL clarification replies append the exact numbered candidate dictionaries in stored order, after the model's question.
- The same response can propose at most eight memory updates (`unit/unit_id`, `faction/faction_id`, `detachment/detachment_id`, `topic`, `conditions`). User values must be verbatim substrings of the current input; SQL identities must match unique values from recognized entity columns of an untruncated result. Arbitrary aliases, statistics and multi-entity lists do not populate SQL memory. Values carry the current `source_run_id`; changed names/IDs clear an old unmatched identity partner. Grounding verifies provenance, not the semantic suitability of every chosen slot.
- `finalize` merges validated scope updates and commits one Human/AI pair with stable `<run_id>:user/assistant` IDs. Repeating the commit replaces those IDs rather than duplicating them. The committed window is trimmed to eight full turns and approximately 5,000 tokens, using the same policy as preparation. Clarification/unsupported and valid partial replies are committed; error/cancelled outcomes and missing replies are not. No extra summarization or memory-extraction model call is added.

Direct graph calls propagate generation failures, malformed response JSON, invalid references or ungrounded memory updates and retain the last successful checkpoint. No incomplete reply or new memory is committed. The turn runner converts those exceptions and Ctrl-C into terminal run results. The evidence-only graph still leaves `answer / outcome` unset. See the [Agentic RAG Spec](agentic-rag-spec.md).

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
- Events contain schema version, session/run IDs, sequence, UTC timestamp, event/node, span and parent IDs. They cover `run.started`, node `step.started/finished/failed`, `route.selected`, `query.refined`, `retrieval.completed` and exactly one `run.finished` for a controlled turn. Node starts snapshot their actual State input; finishes snapshot updates, including history trimming/overwrite, planned queries, bounded retriever results, PDF metadata, SQL provenance, answer, citations and memory commits. These are **graph-node** inputs, not a recording of the exact model HTTP requests.
- Durations use a monotonic clock and measure observed node lifecycles, including stream/trace/display overhead; run time includes cold model initialization. LLM events record observed SDK-call duration, including any SDK-internal retries. SQL's non-LLM internal phases, raw model outputs, repair SQL before/after snapshots and rulebook intermediate candidate rankings remain unavailable.
- The writer flushes each complete JSON line. It does not promise power-loss durability. Write/open/close failures emit a visible `trace.failed` warning and mark the returned trace incomplete while preserving a valid answer. Error records retain exception type, validation locations and stack file/function/line locations, omitting raw provider messages, locals and source lines. Known configured API keys and credential fields are redacted; SDK clients and authentication headers are never serialized from runtime context.
- `--inspect` reads saved records without building/resuming a graph or calling a model/SQL. It shows lifecycle, route and result; `--sources` resolves actual cited chunks or executed SQL, bindings, full bounded rows and separate prompt-clipping flags. `--json` exposes full recorded node snapshots. An incomplete final JSON fragment is skipped with a warning; a run without `run.finished` is `interrupted`. Corrupt middle records or mixed run IDs/sequence are rejected. This is replay of recorded facts, not a rerun of the original experiment.

The lower-level graph still propagates exceptions. The runner's returned outcome is authoritative for failed/cancelled turns: its returned State copy clears answer/citations; the underlying checkpoint retains the last successful node for debugging. Cancellation does not forcibly terminate synchronous provider/model calls; it may wait for an in-flight call to return before the stream stops. No new turn is submitted during that wait. Once `finalize` starts, the memory commit completes and the committed outcome wins over a late cancellation. Automatic checkpoint rotation for long sessions and detailed retriever instrumentation remain pending.

### Textual TUI

```bash
uv run python -m ui.tui
uv run python -m ui.tui --model gemini-3.1-flash-lite
uv run python -m ui.tui --inspect data/traces/SESSION/RUN.jsonl
```

`ui/tui.py` uses a Textual thread worker to call the same runner as the CLI. Only redacted trace messages cross into the UI through thread-safe `post_message`; stdout/stderr output is not parsed. One turn runs at a time. Input, New and the turn selector are disabled until the worker returns. Esc requests cancellation; Ctrl+Q requests cancellation and quits after the worker returns.

The chat is paired with an inspector showing session/run/turn/model, route and `both` dependency, observed step/total timings, SQL strategy/row count, rulebook count, clipping indicators, trace status and failed node. LLM child rows show each actual SDK call; metadata shows per-turn and session calls/tokens/coverage. Select a turn to load its retained JSONL on demand; Enter on a step opens its complete redacted input/output in Payload. Select `run.finished` for the complete final state, usage breakdown, error/stack locations, context and clarification metadata. Sources resolves cited SQL/parameters/rows and rulebook chunk IDs/pages/text/scores through `source_details`. The inspector can be toggled with Ctrl+T or `/trace`; below 100 columns it replaces the chat view. Ctrl+S or `/sources [ID]` opens sources.

`/new` starts a fresh session, deletes the old in-process checkpoints and clears the UI's run list, while retaining disk traces. The chat rendering buffer is limited to 1,000 lines; only the selected run's full events/state are retained in UI memory. `--no-trace` keeps current inspection but disables access to earlier turn details. `--inspect` is read only: it creates no graph/client and executes no model or SQL calls. It also displays interrupted traces and reader warnings. Live replies appear as complete answers; no answer-token streaming or retriever-internal timings are implied.

Headless UI tests use Textual's `run_test`/Pilot with `unittest.IsolatedAsyncioTestCase`, real graph orchestration and offline model/retriever stubs; no real API calls are required. See [Textual workers](https://textual.textualize.io/guide/workers/) and [testing](https://textual.textualize.io/guide/testing/).

### LLM calls and provider token usage

`llm/instrumentation.py` wraps `generate_content` without changing its response or adding retries. Plan, dependent rulebook-query refinement, response generation, SQL routing/generation/repair and the legacy fixed-function entry use it. Each invocation emits `llm.started` and one `llm.finished`/`llm.failed`, with a unique call ID parented to its graph node. Events retain purpose, requested model, provider, monotonic duration, response ID/model version when supplied, and usage. Provider error text, credentials, prompts and raw responses are not copied into LLM metadata events. Usage is captured before JSON/citation validation, so invalid structured output still counts. `llm_calls` counts SDK invocations including failed calls; it does not claim to count HTTP retries hidden inside the SDK or evaluation-judge calls.

Provider usage preserves nullable `prompt_token_count`, `candidates_token_count`, `thoughts_token_count`, `cached_content_token_count`, `tool_use_prompt_token_count` and `total_token_count`. Values are reported, never estimated; cached tokens are not added to prompt tokens and the provider's total is not reconstructed from a formula. API errors or an absent usage response have `usage: null` at the call level. A missing field stays null; an explicitly reported zero stays zero.

`run.finished` and the runner result expose `llm_calls` and a `usage` summary. Each token total is null unless every observed call reports that field and the instrumented scope is complete. `known_totals` holds only reported subtotals, `reported_calls` gives coverage per field, and `total_calls` is the observed-call denominator. `complete` requires complete prompt/candidate/total coverage; optional thinking/cache/tool fields may remain null. A question-only injected SQL callable is opaque when executed: `instrumentation_complete=false` marks that boundary, while retaining counts and known usage from instrumented calls. There are no usage counters in checkpointed `AgentState` or future prompts.

CLI/TUI also reduce run summaries into session totals, keyed by run ID; history selection never adds the same run again. `/new` resets totals. `--no-trace` still records live statistics in memory. Old traces stay unavailable and are not estimated from node counts. Evaluation's trace adapter passes through the same fields. During cancellation the runner waits for in-flight work and drains retained LLM events before writing `run.finished`, preserving any returned usage even when no answer is committed. A killed process can leave only `llm.started`; its consumed tokens remain unknown.

## Build and Simple Verification

Install `uv` and GNU Make 4.3 or newer, then configure `GOOGLE_API_KEY` in `.env`
(copy `.env.example`) or in the environment. From the repository root:

```bash
make prepare
```

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
uv run --group eval python -m evaluation.run --dataset all --ragas
```

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
