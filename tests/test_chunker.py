"""Small unit tests for document-aware chunk construction."""

import json
import re
import unittest
from pathlib import Path

from ingestion.chunker import PictureLink, apply_corrections, chunk_document, load_corrections


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


class ChunkerTests(unittest.TestCase):
    def test_applies_source_backed_visual_text_correction(self):
        chunks = [{
            "chunk_id": "core-rules-grenade-001-001",
            "text": "GRENADE\n\nCORE - WARGEAR STRATAGEM\n\nWHEN: Your Shooting phase.",
            "pdf_page_numbers": [42],
            "token_count": 8,
        }]
        correction = {
            "correction_id": "core-rules-page-42-grenade-cost",
            "chunk_id": "core-rules-grenade-001-001",
            "pdf_page_number": 42,
            "operation": "insert_after",
            "anchor": "CORE - WARGEAR STRATAGEM",
            "text": "COST: 1CP",
            "source_evidence": "Visible label on PDF page 42.",
            "reason": "The layout association was lost during parsing.",
        }

        corrected = apply_corrections(chunks, [correction], count_tokens=count_words)
        self.assertIn("CORE - WARGEAR STRATAGEM\n\nCOST: 1CP", corrected[0]["text"])
        self.assertEqual(corrected[0]["token_count"], count_words(corrected[0]["text"]))
        self.assertEqual(
            apply_corrections(corrected, [correction], count_tokens=count_words),
            corrected,
        )

    def test_project_correction_matches_canonical_grenade_chunk(self):
        root = Path(__file__).resolve().parents[1]
        corrections = load_corrections(root / "data/processed/Core Rules.corrections.json")
        chunks = [
            json.loads(line)
            for line in (root / "data/processed/Core Rules.chunks.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        grenade = next(
            chunk for chunk in chunks
            if chunk["chunk_id"] == "core-rules-grenade-001-001"
        )

        self.assertEqual(len(corrections), 1)
        self.assertEqual(corrections[0]["pdf_page_number"], 42)
        self.assertIn(corrections[0]["text"], grenade["text"])

    def test_keeps_section_context_and_filters_page_furniture(self):
        document = {
            "body": {"children": [
                {"$ref": "#/texts/0"}, {"$ref": "#/texts/1"},
                {"$ref": "#/texts/2"}, {"$ref": "#/groups/0"},
                {"$ref": "#/pictures/0"},
            ]},
            "texts": [
                {"self_ref": "#/texts/0", "label": "section_header", "level": 1,
                 "text": "Core Rules", "prov": [{"page_no": 1}]},
                {"self_ref": "#/texts/1", "label": "section_header", "level": 2,
                 "text": "Movement", "prov": [{"page_no": 2}]},
                {"self_ref": "#/texts/2", "label": "text", "text": "Models move in units.",
                 "prov": [{"page_no": 2}]},
                {"self_ref": "#/texts/3", "label": "list_item", "text": "Measure from base.",
                 "prov": [{"page_no": 2}]},
            ],
            "groups": [{"self_ref": "#/groups/0", "label": "list", "children": [{"$ref": "#/texts/3"}]}],
            "pictures": [{"self_ref": "#/pictures/0", "label": "picture", "prov": [{"page_no": 2}]}],
            "tables": [],
        }
        pictures = {"#/pictures/0": PictureLink("#/pictures/0", "images/move.png", {2})}
        chunks = chunk_document(document, source="rules.pdf", pictures=pictures, count_tokens=count_words)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["section_path"], ["Core Rules", "Movement"])
        self.assertIn("Models move in units.", chunks[0]["text"])
        self.assertIn("Measure from base.", chunks[0]["text"])
        self.assertEqual(chunks[0]["picture_ids"], ["#/pictures/0"])
        self.assertEqual(chunks[0]["image_paths"], ["images/move.png"])

    def test_long_unit_creates_parts_with_shared_logical_id(self):
        sentences = " ".join(f"Rule sentence number {number}." for number in range(1, 25))
        document = {
            "body": {"children": [{"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}]},
            "texts": [
                {"self_ref": "#/texts/0", "label": "section_header", "level": 1,
                 "text": "Attacks", "prov": [{"page_no": 21}]},
                {"self_ref": "#/texts/1", "label": "text", "text": sentences,
                 "prov": [{"page_no": 21}]},
            ],
            "groups": [], "pictures": [], "tables": [],
        }
        chunks = chunk_document(
            document, source="rules.pdf", count_tokens=count_words,
            max_tokens=20, overlap_tokens=4,
        )

        self.assertGreater(len(chunks), 1)
        self.assertEqual({chunk["logical_unit_id"] for chunk in chunks}, {chunks[0]["logical_unit_id"]})
        self.assertEqual([chunk["part_index"] for chunk in chunks], list(range(1, len(chunks) + 1)))
        self.assertTrue(all(chunk["part_count"] == len(chunks) for chunk in chunks))
        self.assertTrue(all(chunk["token_count"] <= 20 for chunk in chunks))

    def test_heading_case_changes_do_not_reuse_chunk_ids(self):
        document = {
            "body": {"children": [
                {"$ref": "#/texts/0"}, {"$ref": "#/texts/1"},
                {"$ref": "#/texts/2"}, {"$ref": "#/texts/3"},
            ]},
            "texts": [
                {"self_ref": "#/texts/0", "label": "section_header", "level": 1,
                 "text": "OBJECTIVE MARKERS", "prov": [{"page_no": 58}]},
                {"self_ref": "#/texts/1", "label": "text", "text": "First rule.",
                 "prov": [{"page_no": 58}]},
                {"self_ref": "#/texts/2", "label": "section_header", "level": 1,
                 "text": "Objective Markers", "prov": [{"page_no": 59}]},
                {"self_ref": "#/texts/3", "label": "text", "text": "Second rule.",
                 "prov": [{"page_no": 59}]},
            ],
            "groups": [], "pictures": [], "tables": [],
        }
        chunks = chunk_document(document, source="rules.pdf", count_tokens=count_words)

        self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), len(chunks))


if __name__ == "__main__":
    unittest.main()
