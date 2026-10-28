.DEFAULT_GOAL := prepare
.DELETE_ON_ERROR:
.ONESHELL:
.SHELLFLAGS := -eu -c

RUN := uv run --no-sync python
BUILD_INPUTS := Makefile pyproject.toml uv.lock
PARSED := data/processed/Core\ Rules.json \
	data/processed/Core\ Rules.md \
	data/processed/images/Core\ Rules/manifest.json
INDEX := data/indexes/core_rules
INDEX_FILES := $(INDEX)/chroma/chroma.sqlite3 $(INDEX)/bm25.pkl \
	$(INDEX)/bm25_chunk_ids.json $(INDEX)/index_manifest.json

.PHONY: prepare sync inputs parse chunk index check

prepare: check

sync:
	uv sync --locked --group eval --managed-python

inputs: sync data/pdf/Core\ Rules.pdf data/sql/wahadb.sqlite \
	data/sql/SQL_AGENT_DATA_DICTIONARY.md data/processed/Core\ Rules.corrections.json
	@$(RUN) - <<'PY'
	import os
	import sqlite3
	from pathlib import Path
	from dotenv import load_dotenv
	load_dotenv(".env")
	key = os.getenv("GOOGLE_API_KEY", "").strip()
	if not key or key == "your_google_ai_api_key_here": raise SystemExit("Set GOOGLE_API_KEY in .env or the environment before running make prepare.")
	Path("data/sql/SQL_AGENT_DATA_DICTIONARY.md").read_text(encoding="utf-8")
	with sqlite3.connect("file:data/sql/wahadb.sqlite?mode=ro", uri=True) as connection: connection.execute("SELECT Id FROM datasheets LIMIT 1").fetchone()
	print("Checked API key configuration, SQLite, and data dictionary.")
	PY

parse: $(PARSED)

$(PARSED) &: data/pdf/Core\ Rules.pdf ingestion/pdf_parser.py $(BUILD_INPUTS) | inputs
	$(RUN) -m ingestion.pdf_parser "data/pdf/Core Rules.pdf"

chunk: data/processed/Core\ Rules.chunks.jsonl

data/processed/Core\ Rules.chunks.jsonl: $(PARSED) \
	data/processed/Core\ Rules.corrections.json ingestion/chunker.py $(BUILD_INPUTS) | inputs
	$(RUN) -m ingestion.chunker "data/processed/Core Rules.json" \
		--corrections "data/processed/Core Rules.corrections.json"

index: $(INDEX_FILES)

$(INDEX_FILES) &: data/processed/Core\ Rules.chunks.jsonl ingestion/build_index.py $(BUILD_INPUTS) | inputs
	$(RUN) -m ingestion.build_index "data/processed/Core Rules.chunks.jsonl" --recreate

check: index
	$(RUN) -m evaluation.dataset_loader
	$(RUN) -m retrieval.rulebook_retriever "How does a wound roll work?" > /dev/null
	printf 'Preparation complete: CLI/TUI and evaluation dependencies are ready.\n'
