"""Parse PDFs with Docling, retaining structure and page provenance."""

import argparse
import json
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument, ImageRefMode
from langchain_core.documents import Document


def parse_pdf(path: str | Path, *, ocr: bool = False) -> DoclingDocument:
    """Return the structured document, including tables and item provenance.

    OCR is optional because the rulebook has an embedded text layer. Docling
    may download layout/table models on the first conversion. No LLM key is
    needed. Failed or partial conversions raise instead of hiding lost pages.
    """
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    options = PdfPipelineOptions(
        do_ocr=ocr, do_table_structure=True,
        generate_picture_images=True, images_scale=2.0,
    )
    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options)
    })
    result = converter.convert(pdf_path)
    if result.status != ConversionStatus.SUCCESS:
        raise RuntimeError(f"PDF conversion was incomplete: {result.status}")
    return result.document


def to_langchain_pages(document: DoclingDocument, source: str) -> list[Document]:
    """Create page Markdown views; retain the original JSON for later chunking.

    Page numbers are physical PDF positions, not printed rulebook labels.
    These page views do not replace the full Docling structure.
    """
    return [
        Document(
            page_content=document.export_to_markdown(page_no=number),
            metadata={"source": source, "page": number - 1,
                      "page_number": number},
        )
        for number in sorted(document.pages)
    ]


def export_images(
    document: DoclingDocument, source: str, output_dir: Path,
) -> Path:
    """Save detected picture crops and a manifest; no image understanding API.

    Paths in the manifest are relative to the manifest directory. All detected
    pictures are exported, including illustrations and decorative elements.
    """
    image_dir = output_dir / "images" / Path(source).stem
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, picture in enumerate(document.pictures, start=1):
        image = picture.get_image(document)
        if image is None:
            raise ValueError(f"No image data for {picture.self_ref}; reparse the PDF.")
        filename = f"picture-{index:04d}.png"
        image.save(image_dir / filename, format="PNG")
        records.append({
            "picture_id": picture.self_ref,
            "source": source,
            "path": filename,
            "page_numbers": sorted({p.page_no for p in picture.prov}),
            "provenance": [p.model_dump(mode="json") for p in picture.prov],
        })
    manifest = image_dir / "manifest.json"
    manifest.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to the rulebook PDF")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR for image text")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    document = parse_pdf(args.path, ocr=args.ocr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / args.path.stem
    manifest = export_images(document, str(args.path), args.output_dir)
    document.save_as_json(stem.with_suffix(".json"), image_mode=ImageRefMode.PLACEHOLDER)
    document.save_as_markdown(stem.with_suffix(".md"))
    print(f"Parsed {len(document.pages)} pages.")
    print(f"Saved structured JSON and Markdown in {args.output_dir}")
    print(f"Exported {len(document.pictures)} pictures; manifest: {manifest}")


if __name__ == "__main__":
    main()
