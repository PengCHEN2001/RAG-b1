"""Small unit tests for document-aware chunk construction."""

import re
import unittest

from ingestion.chunker import PictureLink, chunk_document


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


class ChunkerTests(unittest.TestCase):
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
