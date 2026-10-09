# Project Guide

## Overview

RAG-b1 is a Warhammer 40K question-answering project. It will use the core rulebook PDF for game rules and the SQLite database for unit data.

## Project Structure

```text
RAG-b1/
├── data/
│   ├── pdf/
│   │   └── Core Rules.pdf
│   ├── sql/
│   │   └── wahadb.sqlite
│   ├── processed/
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
├── tests/
│   ├── test_pdf_parser.py
│   └── test_chunker.py
├── docs/
│   ├── journal.md
│   └── project-guide.md
├── .env.example
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

## Chunking Design

Implemented in `ingestion/chunker.py`.

- Input: `data/processed/Core Rules.json`. Markdown is only a human-readable preview.
- Resolve Docling `$ref` values in `body.children` to preserve the original document order.
- Use `section_header` values to build `section_path`. A heading and its body text, lists, and small tables form one logical unit, even when they cross PDF pages.
- Exclude page headers, footers, and empty elements. Pictures are not embedded as text; their IDs, paths, and pages remain attached as metadata.
- Use document-aware recursive splitting only when a logical unit is too long: target 300–400 BGE tokens, maximum 450 tokens, split at paragraph or sentence boundaries, with a 60-token overlap only between parts of the same long unit.
- Output: `data/processed/Core Rules.chunks.jsonl`. The current rulebook produces 212 chunks.

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

## Build and Simple Verification

```bash
uv sync
uv run python -m ingestion.pdf_parser "data/pdf/Core Rules.pdf"
uv run python -m ingestion.chunker "data/processed/Core Rules.json"
uv run python -m ingestion.build_index "data/processed/Core Rules.chunks.jsonl" --recreate
uv run python -m unittest tests.test_pdf_parser tests.test_chunker -v
```

Add `--ocr` to the parser command when OCR is needed. The first run may download Docling model files.
The first chunking and indexing run also downloads `BAAI/bge-base-en-v1.5`.
