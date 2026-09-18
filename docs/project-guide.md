# Project Guide

## Project Structure

```text
RAG-b1/
├── data/
│   ├── pdf/
│   │   └── Core Rules.pdf
│   ├── sql/
│   │   └── wahadb.sqlite
│   └── processed/
│       ├── Core Rules.json
│       ├── Core Rules.md
│       └── images/Core Rules/
│           ├── picture-*.png
│           └── manifest.json
├── ingestion/
│   ├── __init__.py
│   └── pdf_parser.py
├── tests/
│   └── test_pdf_parser.py
├── docs/
│   ├── journal.md
│   └── project-guide.md
├── .env.example
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

## Overview

RAG-b1 is a Warhammer 40K question-answering project. It will use the core rulebook PDF for game rules and the SQLite database for unit data. The current implementation covers PDF processing.

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

## Output

The processed files are stored in `data/processed/`:

- `Core Rules.json`
- `Core Rules.md`
- `images/Core Rules/*.png`
- `images/Core Rules/manifest.json`

The current conversion processed 60 pages and exported 148 images.

## Usage

```bash
uv sync
uv run python -m ingestion.pdf_parser "data/pdf/Core Rules.pdf"
uv run python -m unittest discover -s tests -v
```

Add `--ocr` to the parser command when OCR is needed. The first run may download Docling model files.
