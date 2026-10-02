# Development Journal

## 2026-09-18 — PDF Parser and Image Export

### `ingestion/pdf_parser.py`

- Added the `ingestion` package and implemented PDF parsing with Docling. The parser extracts text, headings, tables, page metadata, and images.
- Designed JSON as the structured source for future chunking and Markdown as a human-readable preview. Images are stored as PNG files, with a manifest linking picture IDs to their source and page metadata.
- Added optional OCR, LangChain page conversion, image export, and conversion error handling. The 60-page rulebook produced JSON, Markdown, 148 PNG images, and a manifest in `data/processed/`.
- Run with `uv run python -m ingestion.pdf_parser "data/pdf/Core Rules.pdf"`.

### `tests/test_pdf_parser.py`

- Added basic tests for image export, metadata preservation, missing files or image data, and incomplete conversions.
- All tests passed successfully.
- Run with `uv run python -m unittest discover -s tests -v`.

## 2026-10-02 — Fixed-Function SQLite Retriever

### `sql_retriver/functions.py`

- Added a function registry containing each tool's description, parameter schema, and Python callable. `get_function_declarations()` exposes only the serializable descriptions and schemas to the model.
- Implemented `get_unit_all_metadata(d_id)` to query basic unit metadata and faction information. The builder returns SQL with named placeholders and a parameter dictionary, preserving string IDs and leading zeros without interpolating values into SQL.
- The initial tool covers unit ID, name, faction, role, lore, and reference link. Model stats, weapons, abilities, and points require additional query functions.

### `sql_retriver/fixed_func.py`

- Implemented `sql_retrieve_fixed_funcs()` to ask Gemini for a structured function selection, validate it, and execute the selected query through Python's built-in `sqlite3` module.
- Defined a JSON response schema with `name` and `arguments`. A null name with empty arguments represents a request that needs entity resolution or another tool. The prompt requires an explicitly supplied or previously resolved datasheet ID rather than a guessed ID.
- Added validation for registered function names, required arguments, unexpected arguments, and non-empty string values. Responses are parsed from `response.text` with `json.loads()`.
- Added `make_sql_query()` to open `data/sql/wahadb.sqlite` in read-only mode, bind named parameters, and fetch rows before closing the connection. Retrieval returns a list of row tuples, an empty list for no matching records, or `None` when no function is selected.
- Explicitly disabled SDK automatic function calling (AFC) in `generate_content()`, since the application validates and dispatches the returned JSON itself. This avoids the SDK warning about direct AFC use in that method.

### `tests/test_sql_retriver.py`

- Added eight tests using a temporary SQLite database and mocked model responses, covering parameter binding, leading-zero preservation, SQL injection strings treated as values, read-only execution, invalid selections, abstention, unmatched IDs, and empty model responses.
- Verified the full selection-to-query flow without API credentials. An SDK-level test replaces the remote request and confirms that AFC is disabled and its warning is not emitted.
- Added project-root path setup for direct execution of the test file. The current `from functions import ...` convention also requires `sql_retriver` on the Python import path.
- All eight tests passed. Run from the project root with `PYTHONPATH=sql_retriver uv run python ./tests/test_sql_retriver.py`. No live Gemini API request was made during this verification.
