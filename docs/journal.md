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
