import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Direct execution puts tests/ on the import path rather than the project root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google import genai
from google.genai import types

from sql_retriver.fixed_func import sql_retrieve_fixed_funcs, make_sql_query, parse_function_call, response_schema
from sql_retriver.functions import get_unit_all_metadata


class FixedFunctionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        db_path = Path(directory.name) / "test.sqlite"
        con = sqlite3.connect(db_path)
        try:
            con.execute("CREATE TABLE datasheets (Id, name, factionId, role, legend, link)")
            con.execute("CREATE TABLE factions (Id, name)")
            con.execute("INSERT INTO datasheets VALUES (?, ?, ?, ?, ?, ?)",
                        ("000000001", "Warboss", "ORK", "", "", ""))
            con.execute("INSERT INTO factions VALUES (?, ?)", ("ORK", "Orks"))
            con.commit()
        finally:
            con.close()
        db_patch = patch("sql_retriver.fixed_func.DB_PATH", db_path)
        db_patch.start()
        self.addCleanup(db_patch.stop)

    def test_string_id_and_parameterized_query(self):
        sql, params = get_unit_all_metadata("000000001")
        self.assertNotIn("000000001", sql)
        self.assertEqual(params, {"d_id": "000000001"})
        self.assertEqual(make_sql_query(sql, params)[0][3], "Orks")
        unsafe_id = "000000001' OR 1=1 --"
        self.assertEqual(make_sql_query(*get_unit_all_metadata(unsafe_id)), [])

    def test_query_connection_is_read_only(self):
        with self.assertRaises(sqlite3.OperationalError):
            make_sql_query("DELETE FROM datasheets", {})
        self.assertEqual(len(make_sql_query(*get_unit_all_metadata("000000001"))), 1)

    def test_invalid_model_selections(self):
        selections = [
            [],
            {"name": "unknown", "arguments": {}},
            {"name": "get_unit_all_metadata", "arguments": {}},
            {"name": "get_unit_all_metadata", "arguments": {"d_id": 1}},
            {"name": "get_unit_all_metadata", "arguments": {"d_id": " "}},
            {"name": "get_unit_all_metadata", "arguments": {"d_id": "1", "sql": "SELECT 1"}},
            {"name": None, "arguments": {"d_id": "1"}},
        ]
        for selection in selections:
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                parse_function_call(json.dumps(selection))

    def test_missing_id_can_abstain(self):
        self.assertIsNone(parse_function_call('{"name": null, "arguments": {}}'))
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(
            text='{"name": null, "arguments": {}}'
        )
        with patch("sql_retriver.fixed_func.make_sql_query") as execute:
            self.assertIsNone(sql_retrieve_fixed_funcs("Get metadata for Warboss", client=client))
            execute.assert_not_called()

    def test_model_response_is_parsed_without_credentials(self):
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(
            text='{"name":"get_unit_all_metadata","arguments":{"d_id":"000000001"}}'
        )
        rows = sql_retrieve_fixed_funcs("Get metadata for datasheet 000000001", client=client)
        self.assertEqual(rows, [("000000001", "Warboss", "ORK", "Orks", "", "", "")])
        config = client.models.generate_content.call_args.kwargs["config"]
        self.assertIn("Get a unit's basic metadata", config["system_instruction"])
        self.assertEqual(types.GenerateContentConfig(**config).response_json_schema,
                         response_schema)

    def test_unmatched_id_returns_empty_rows(self):
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(
            text='{"name":"get_unit_all_metadata","arguments":{"d_id":"missing"}}'
        )
        self.assertEqual(sql_retrieve_fixed_funcs("Get metadata for datasheet missing", client=client), [])

    def test_sdk_skips_afc_without_network_calls(self):
        response = types.GenerateContentResponse(candidates=[
            types.Candidate(content=types.Content(role="model", parts=[
                types.Part(text='{"name": null, "arguments": {}}'),
            ])),
        ])
        # Exercise the installed SDK while replacing only its remote request.
        with genai.Client(api_key="test-placeholder") as client:
            with patch.object(client.models, "_generate_content", return_value=response) as remote:
                with patch("google.genai.models.logger.warning") as warning:
                    self.assertIsNone(sql_retrieve_fixed_funcs("Get metadata", client=client))
            remote.assert_called_once()
            warning.assert_not_called()
            config = remote.call_args.kwargs["config"]
            self.assertTrue(config.automatic_function_calling.disable)

    def test_empty_model_response(self):
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(text=None)
        with self.assertRaises(ValueError):
            sql_retrieve_fixed_funcs("Get metadata", client=client)


if __name__ == "__main__":
    unittest.main()
