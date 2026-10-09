"""Create document-aware, source-traceable chunks from a Docling JSON export."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_TOKENIZER_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_TARGET_TOKENS = 350
DEFAULT_MAX_TOKENS = 450
DEFAULT_OVERLAP_TOKENS = 60


@dataclass
class ContentBlock:
    """A text-bearing Docling element, with enough data for source tracing."""

    text: str
    ref: str
    pages: set[int]
    kind: str
    source_start: int = 0
    source_end: int | None = None

    def __post_init__(self) -> None:
        if self.source_end is None:
            self.source_end = len(self.text)


@dataclass
class PictureLink:
    picture_id: str
    path: str | None
    pages: set[int]


@dataclass
class LogicalUnit:
    identifier: str
    section_path: list[str]
    events: list[ContentBlock | PictureLink] = field(default_factory=list)


def load_document(path: Path) -> dict[str, Any]:
    """Load the JSON emitted by ``DoclingDocument.save_as_json``."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_picture_manifest(path: Path | None) -> dict[str, PictureLink]:
    """Return picture metadata keyed by its Docling ``self_ref``."""
    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8") as file:
        records = json.load(file)
    result: dict[str, PictureLink] = {}
    for record in records:
        image_path = path.parent / record["path"]
        result[record["picture_id"]] = PictureLink(
            picture_id=record["picture_id"],
            path=str(image_path),
            pages=set(record.get("page_numbers", [])),
        )
    return result


def _reference_index(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index all Docling nodes by their JSON pointer-like ``self_ref``."""
    index: dict[str, dict[str, Any]] = {}
    for collection in ("texts", "tables", "pictures", "groups"):
        for item in document.get(collection, []):
            if "self_ref" in item:
                index[item["self_ref"]] = item
    return index


def _pages(item: dict[str, Any]) -> set[int]:
    return {
        provenance["page_no"]
        for provenance in item.get("prov", [])
        if provenance.get("page_no") is not None
    }


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _table_to_text(table: dict[str, Any]) -> str:
    """Convert Docling's grid into compact, readable table text.

    A spanning cell appears repeatedly in Docling's grid.  Keeping only the
    occurrence that starts at the current row and column avoids duplicated
    header text while retaining the table as one semantic block.
    """
    rows: list[str] = []
    for row_index, row in enumerate(table.get("data", {}).get("grid", [])):
        cells: list[str] = []
        for column_index, cell in enumerate(row):
            if not cell:
                continue
            if (cell.get("start_row_offset_idx") != row_index
                    or cell.get("start_col_offset_idx") != column_index):
                continue
            text = _normalise_text(cell.get("text", ""))
            if text:
                cells.append(text)
        if cells:
            rows.append(" | ".join(cells))
    return "Table:\n" + "\n".join(rows) if rows else ""


def _expand_node(
    reference: str,
    index: dict[str, dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    """Yield leaves in visual order, resolving groups and their child refs."""
    item = index.get(reference)
    if item is None:
        return
    if item.get("label") == "list" or item.get("name") == "list":
        for child in item.get("children", []):
            child_ref = child.get("$ref")
            if child_ref:
                yield from _expand_node(child_ref, index)
        return
    yield item


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "section"


def _make_unit_id(source: str, section_path: list[str], occurrence: int) -> str:
    prefix = _slug(Path(source).stem)
    path = "-".join(_slug(part) for part in section_path) or "front-matter"
    return f"{prefix}-{path}-{occurrence:03d}"


def build_logical_units(
    document: dict[str, Any],
    *,
    source: str,
    pictures: dict[str, PictureLink] | None = None,
) -> list[LogicalUnit]:
    """Group body elements under their current Docling heading hierarchy."""
    index = _reference_index(document)
    pictures = pictures or {}
    section_path: list[str] = []
    units: list[LogicalUnit] = []
    current: LogicalUnit | None = None
    occurrences: dict[str, int] = {}

    def start_unit() -> LogicalUnit:
        # Heading case varies in the PDF (e.g. ``OBJECTIVE MARKERS`` versus
        # ``Objective Markers``), so use the same normalised form as IDs.
        key = " > ".join(_slug(part) for part in section_path) or "front-matter"
        occurrences[key] = occurrences.get(key, 0) + 1
        return LogicalUnit(
            identifier=_make_unit_id(source, section_path, occurrences[key]),
            section_path=section_path.copy(),
        )

    def flush() -> None:
        nonlocal current
        if current and any(isinstance(event, ContentBlock) for event in current.events):
            units.append(current)
        current = None

    for child in document.get("body", {}).get("children", []):
        reference = child.get("$ref")
        if not reference:
            continue
        for item in _expand_node(reference, index):
            label = item.get("label")
            if label == "section_header":
                flush()
                level = item.get("level") or 1
                heading = _normalise_text(item.get("text", ""))
                if not heading:
                    continue
                section_path = section_path[:max(0, level - 1)] + [heading]
                continue
            if label in {"page_header", "page_footer"}:
                continue
            if current is None:
                current = start_unit()
            if reference.startswith("#/pictures/") or item.get("label") == "picture":
                metadata = pictures.get(item["self_ref"])
                current.events.append(metadata or PictureLink(
                    picture_id=item["self_ref"], path=None, pages=_pages(item),
                ))
                continue
            if reference.startswith("#/tables/") or label == "table":
                text = _table_to_text(item)
                kind = "table"
            else:
                text = _normalise_text(item.get("text", ""))
                kind = "list" if label == "list_item" else "text"
                if kind == "list" and text and not text.startswith(("- ", "• ", "■ ")):
                    text = f"- {text}"
            if text:
                current.events.append(ContentBlock(
                    text=text, ref=item["self_ref"], pages=_pages(item), kind=kind,
                ))
    flush()
    return units


def _render_text(section_path: list[str], blocks: list[ContentBlock]) -> str:
    title = " > ".join(section_path)
    content = "\n\n".join(block.text for block in blocks)
    return f"{title}\n\n{content}".strip() if title else content


def _split_block(
    block: ContentBlock,
    max_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[ContentBlock]:
    """Split a single oversized block on sentences, then whitespace if needed."""
    if count_tokens(block.text) <= max_tokens:
        return [block]
    sentences = re.split(r"(?<=[.!?])\s+", block.text)
    fragments: list[ContentBlock] = []
    current = ""
    cursor = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip()
        if current and count_tokens(candidate) > max_tokens:
            start = block.text.find(current, cursor)
            fragments.append(ContentBlock(
                text=current, ref=block.ref, pages=block.pages, kind=block.kind,
                source_start=block.source_start + max(start, 0),
                source_end=block.source_start + max(start, 0) + len(current),
            ))
            cursor = max(start, 0) + len(current)
            current = sentence
        else:
            current = candidate
    if current:
        start = block.text.find(current, cursor)
        fragments.append(ContentBlock(
            text=current, ref=block.ref, pages=block.pages, kind=block.kind,
            source_start=block.source_start + max(start, 0),
            source_end=block.source_start + max(start, 0) + len(current),
        ))
    if len(fragments) == 1 and count_tokens(fragments[0].text) > max_tokens:
        words = block.text.split()
        fragments = []
        current_words: list[str] = []
        for word in words:
            candidate = " ".join([*current_words, word])
            if current_words and count_tokens(candidate) > max_tokens:
                text = " ".join(current_words)
                start = block.text.find(text, cursor)
                fragments.append(ContentBlock(
                    text=text, ref=block.ref, pages=block.pages, kind=block.kind,
                    source_start=block.source_start + max(start, 0),
                    source_end=block.source_start + max(start, 0) + len(text),
                ))
                cursor = max(start, 0) + len(text)
                current_words = [word]
            else:
                current_words.append(word)
        if current_words:
            text = " ".join(current_words)
            start = block.text.find(text, cursor)
            fragments.append(ContentBlock(
                text=text, ref=block.ref, pages=block.pages, kind=block.kind,
                source_start=block.source_start + max(start, 0),
                source_end=block.source_start + max(start, 0) + len(text),
            ))
    return fragments


def _overlap_block(blocks: list[ContentBlock], overlap_tokens: int,
                   count_tokens: Callable[[str], int]) -> ContentBlock | None:
    """Create a small tail overlap from the latest text block only."""
    if not blocks or overlap_tokens <= 0:
        return None
    last = blocks[-1]
    words = last.text.split()
    tail: list[str] = []
    for word in reversed(words):
        candidate = " ".join(reversed([word, *tail]))
        if tail and count_tokens(candidate) > overlap_tokens:
            break
        tail.insert(0, word)
    text = " ".join(tail)
    if not text:
        return None
    start = last.text.rfind(text)
    return ContentBlock(
        text=text, ref=last.ref, pages=last.pages, kind=last.kind,
        source_start=last.source_start + max(start, 0),
        source_end=last.source_start + max(start, 0) + len(text),
    )


def _chunk_unit(
    unit: LogicalUnit,
    *,
    max_tokens: int,
    overlap_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[tuple[list[ContentBlock], list[PictureLink]]]:
    """Split a logical unit only when required by the token limit."""
    blocks = [event for event in unit.events if isinstance(event, ContentBlock)]
    all_text = _render_text(unit.section_path, blocks)
    if count_tokens(all_text) <= max_tokens:
        return [(blocks, [event for event in unit.events if isinstance(event, PictureLink)])]

    parts: list[tuple[list[ContentBlock], list[PictureLink]]] = []
    current_blocks: list[ContentBlock] = []
    current_pictures: list[PictureLink] = []

    def flush_part(with_overlap: bool) -> None:
        nonlocal current_blocks, current_pictures
        if not current_blocks:
            return
        parts.append((current_blocks, current_pictures))
        overlap = _overlap_block(current_blocks, overlap_tokens, count_tokens) if with_overlap else None
        current_blocks = [overlap] if overlap else []
        current_pictures = []

    for event in unit.events:
        if isinstance(event, PictureLink):
            current_pictures.append(event)
            continue
        # Reserve room for the repeated section path and separators.  The
        # minimum matters for short test limits too; real runs use 450 tokens.
        prefix_budget = max(1, max_tokens - count_tokens(" > ".join(unit.section_path)) - 4)
        for fragment in _split_block(event, prefix_budget, count_tokens):
            candidate = [*current_blocks, fragment]
            if current_blocks and count_tokens(_render_text(unit.section_path, candidate)) > max_tokens:
                flush_part(with_overlap=True)
            current_blocks.append(fragment)
    flush_part(with_overlap=False)
    return parts


def _content_type(blocks: list[ContentBlock]) -> str:
    kinds = {block.kind for block in blocks}
    if kinds == {"table"}:
        return "table"
    return "mixed" if "table" in kinds or "list" in kinds else "text"


def _record_chunk(
    unit: LogicalUnit,
    blocks: list[ContentBlock],
    pictures: list[PictureLink],
    *,
    part_index: int,
    part_count: int,
    source: str,
    count_tokens: Callable[[str], int],
) -> dict[str, Any]:
    text = _render_text(unit.section_path, blocks)
    refs = list(dict.fromkeys([block.ref for block in blocks] + [picture.picture_id for picture in pictures]))
    pages = sorted({page for block in blocks for page in block.pages}
                   | {page for picture in pictures for page in picture.pages})
    return {
        "chunk_id": f"{unit.identifier}-{part_index:03d}",
        "logical_unit_id": unit.identifier,
        "part_index": part_index,
        "part_count": part_count,
        "text": text,
        "content_type": _content_type(blocks),
        "section_path": unit.section_path,
        "source": source,
        "pdf_page_numbers": pages,
        "element_refs": refs,
        "source_spans": [
            {"ref": block.ref, "start": block.source_start, "end": block.source_end}
            for block in blocks
        ],
        "picture_ids": [picture.picture_id for picture in pictures],
        "image_paths": [picture.path for picture in pictures if picture.path],
        "token_count": count_tokens(text),
    }


def chunk_document(
    document: dict[str, Any],
    *,
    source: str,
    count_tokens: Callable[[str], int],
    pictures: dict[str, PictureLink] | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[dict[str, Any]]:
    """Return chunks in document order, retaining section and source metadata."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    units = build_logical_units(document, source=source, pictures=pictures)
    chunks: list[dict[str, Any]] = []
    for unit in units:
        parts = _chunk_unit(
            unit, max_tokens=max_tokens, overlap_tokens=overlap_tokens,
            count_tokens=count_tokens,
        )
        for part_index, (blocks, linked_pictures) in enumerate(parts, start=1):
            chunks.append(_record_chunk(
                unit, blocks, linked_pictures,
                part_index=part_index, part_count=len(parts), source=source,
                count_tokens=count_tokens,
            ))
    return chunks


def _bge_token_counter(model_name: str) -> Callable[[str], int]:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - depends on optional runtime install
        raise RuntimeError("Install sentence-transformers before running the chunker.") from error
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # This tokenizer is used only to count a full logical unit before it is
    # split.  It must not warn merely because that pre-split unit is >512.
    tokenizer.model_max_length = 1_000_000
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Docling JSON export")
    parser.add_argument("--source", default="data/pdf/Core Rules.pdf")
    parser.add_argument("--manifest", type=Path, help="Picture manifest JSON")
    parser.add_argument("--output", type=Path, default=Path("data/processed/Core Rules.chunks.jsonl"))
    parser.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    args = parser.parse_args()

    document = load_document(args.input)
    manifest = args.manifest or args.input.parent / "images" / args.input.stem / "manifest.json"
    chunks = chunk_document(
        document, source=args.source, pictures=load_picture_manifest(manifest),
        count_tokens=_bge_token_counter(args.tokenizer_model),
        max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens,
    )
    count = write_jsonl(chunks, args.output)
    digest = sha256(args.output.read_bytes()).hexdigest()[:12]
    print(f"Wrote {count} chunks to {args.output} (sha256: {digest}).")


if __name__ == "__main__":
    main()
