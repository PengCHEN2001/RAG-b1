"""Tests use Docling data objects without downloading conversion models."""

import tempfile
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from docling.datamodel.base_models import ConversionStatus
from docling_core.types.doc import DoclingDocument, DocItemLabel, ProvenanceItem
from docling_core.types.doc import BoundingBox, Size

from PIL import Image
from docling_core.types.doc import ImageRef

from ingestion.pdf_parser import export_images, parse_pdf, to_langchain_pages


class PDFParserTests(unittest.TestCase):
    def test_image_export_preserves_pixels_and_source(self):
        document = DoclingDocument(name="rules")
        document.add_picture(
            image=ImageRef.from_pil(Image.new("RGB", (20, 10), "red"), dpi=144),
            prov=ProvenanceItem(page_no=3, charspan=(0, 0),
                                bbox=BoundingBox(l=0, t=0, r=10, b=5)),
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = export_images(document, "rules.pdf", Path(directory))
            records = json.loads(manifest.read_text())
            self.assertEqual(records[0]["page_numbers"], [3])
            self.assertEqual(records[0]["picture_id"], document.pictures[0].self_ref)
            self.assertEqual(records[0]["source"], "rules.pdf")
            with Image.open(manifest.parent / records[0]["path"]) as image:
                self.assertEqual(image.size, (20, 10))
                self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))

    def test_missing_image_data_is_reported(self):
        document = DoclingDocument(name="rules")
        document.add_picture()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No image data"):
                export_images(document, "rules.pdf", Path(directory))

    def test_page_views_and_json_preserve_provenance(self):
        document = DoclingDocument(name="rules")
        document.add_page(page_no=1, size=Size(width=300, height=300))
        document.add_page(page_no=2, size=Size(width=300, height=300))
        document.add_text(
            label=DocItemLabel.PARAGRAPH, text="Grenade rule example",
            prov=ProvenanceItem(page_no=1, charspan=(0, 20),
                                bbox=BoundingBox(l=0, t=0, r=100, b=20)),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.json"
            document.save_as_json(path)
            restored = DoclingDocument.load_from_json(path)
        pages = to_langchain_pages(restored, "rules.pdf")
        self.assertEqual(len(pages), 2)
        self.assertIn("Grenade rule example", pages[0].page_content)
        self.assertEqual(pages[1].page_content.strip(), "")
        self.assertEqual(pages[1].metadata["page_number"], 2)
        self.assertEqual(pages[0].metadata["source"], "rules.pdf")
        self.assertEqual(restored.texts[0].prov[0].page_no, 1)

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                parse_pdf(Path(directory) / "missing.pdf")

    @patch("ingestion.pdf_parser.DocumentConverter")
    def test_partial_conversion_is_rejected(self, converter):
        converter.return_value.convert.return_value = SimpleNamespace(
            status=ConversionStatus.PARTIAL_SUCCESS,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pdf"
            path.touch()
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                parse_pdf(path)
