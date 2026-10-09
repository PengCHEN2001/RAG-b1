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
            con.executemany("INSERT INTO datasheets VALUES (?, ?, ?, ?, ?, ?)", [
                ("000000002", "Boyz", "ORK", "Battleline", "", ""),
                ("000000003", "Enemy", "TYR", "Battleline", "", ""),
            ])
            con.execute("CREATE TABLE abilities (Id, name, description, factionId)")
            con.executemany("INSERT INTO abilities VALUES (?, ?, ?, ?)", [
                ("ability1", "Core ability", "Catalog rules", "ORK"),
                ("ability1", "Core ability", "Catalog rules", "TYR"),
            ])
            con.execute("CREATE TABLE datasheets_abilities (DatasheetId, abilityId, name, description)")
            con.executemany("INSERT INTO datasheets_abilities VALUES (?, ?, ?, ?)", [
                ("000000001", "ability1", "", ""),
                ("000000001", "", "Card ability", "Card rules"),
                ("000000001", "ability1", "Override", "Override rules"),
            ])
            con.execute("CREATE TABLE datasheets_keywords (DatasheetId, keyword, model, isFactionKeyword)")
            con.executemany("INSERT INTO datasheets_keywords VALUES (?, ?, ?, ?)", [
                ("000000001", "ORKS", "Warboss", "true"),
                ("000000003", "TYRANIDS", "Enemy", "true"),
            ])
            con.execute("CREATE TABLE stratagems (id, name, type, cpCost, phase, description)")
            con.executemany("INSERT INTO stratagems VALUES (?, ?, ?, ?, ?, ?)", [
                ("strat1", "Test stratagem", "Battle Tactic", "1", "Fight", "Rules"),
                ("strat2", "Enemy stratagem", "Battle Tactic", "2", "Fight", "Enemy rules"),
            ])
            con.execute("CREATE TABLE datasheets_stratagems (DatasheetId, stratagemId)")
            con.executemany("INSERT INTO datasheets_stratagems VALUES (?, ?)", [
                ("000000001", "strat1"), ("000000003", "strat2"),
            ])
            con.execute('CREATE TABLE datasheets_wargear (DatasheetId, name, type, "range", a, bsWs, s, ap, d)')
            con.executemany("INSERT INTO datasheets_wargear VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [
                ("000000001", "Test weapon", "Melee", "Melee", "D6", "2+", "10", "-2", "D3"),
                ("000000003", "Enemy weapon", "Ranged", '24"', "2", "3+", "4", "0", "1"),
            ])
            con.execute("CREATE TABLE detachments (Id, name, factionId)")
            con.executemany("INSERT INTO detachments VALUES (?, ?, ?)", [
                ("det1", "War Horde", "ORK"), ("det2", "No ability", "ORK"),
                ("det3", "Enemy detachment", "TYR"),
            ])
            con.execute("CREATE TABLE detachment_abilities (Id, name, detachmentId)")
            con.executemany("INSERT INTO detachment_abilities VALUES (?, ?, ?)", [
                ("da1", "Detachment rule", "det1"), ("da2", "Enemy rule", "det3"),
            ])
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

    def test_unit_abilities_keep_card_rules_and_deduplicate_catalog(self):
        sql_query = parse_function_call(json.dumps({
            "name": "get_unit_abilities", "arguments": {"d_id": "000000001"},
        }))
        self.assertEqual(set(make_sql_query(*sql_query)), {
            ("Warboss", "ability1", "Core ability", "Catalog rules"),
            ("Warboss", None, "Card ability", "Card rules"),
            ("Warboss", "ability1", "Override", "Override rules"),
        })

    def test_unit_related_queries_do_not_leak_other_unit_rows(self):
        cases = {
            "get_unit_keywords": [("Warboss", "ORKS", "Warboss", "true")],
            "get_unit_stratagems": [("Warboss", "strat1", "Test stratagem", "Battle Tactic", "1", "Fight", "Rules")],
            "get_unit_weapons": [("Warboss", "Test weapon", "Melee", "Melee", "D6", "2+", "10", "-2", "D3")],
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                sql_query = parse_function_call(json.dumps({
                    "name": name, "arguments": {"d_id": "000000001"},
                }))
                self.assertEqual(make_sql_query(*sql_query), expected)

    def test_faction_query_sorting_and_optional_role(self):
        for arguments, expected in [
            ({"faction_id": "ORK"}, [("000000002", "Boyz", "Battleline"), ("000000001", "Warboss", "")]),
            ({"faction_id": "ORK", "role": "Battleline"}, [("000000002", "Boyz", "Battleline")]),
            ({"faction_id": "ORK' OR 1=1 --"}, []),
            ({"faction_id": "ORK", "role": "Battleline' OR 1=1 --"}, []),
        ]:
            with self.subTest(arguments=arguments):
                sql_query = parse_function_call(json.dumps({
                    "name": "get_units_by_faction", "arguments": arguments,
                }))
                self.assertEqual(make_sql_query(*sql_query), expected)

    def test_detachments_keep_entries_without_abilities(self):
        sql_query = parse_function_call(json.dumps({
            "name": "get_detachments_and_abilities", "arguments": {"faction_id": "ORK"},
        }))
        self.assertEqual(set(make_sql_query(*sql_query)), {
            ("det1", "War Horde", "da1", "Detachment rule"),
            ("det2", "No ability", None, None),
        })

    def test_invalid_model_selections(self):
        selections = [
            [],
            {"name": "unknown", "arguments": {}},
            {"name": "get_unit_all_metadata", "arguments": {}},
            {"name": "get_unit_all_metadata", "arguments": {"d_id": 1}},
            {"name": "get_unit_all_metadata", "arguments": {"d_id": " "}},
            {"name": "get_unit_all_metadata", "arguments": {"d_id": "1", "sql": "SELECT 1"}},
            {"name": None, "arguments": {"d_id": "1"}},
            {"name": "get_units_by_faction", "arguments": {"d_id": "1"}},
            {"name": "get_units_by_faction", "arguments": {"faction_id": "ORK", "role": 1}},
            {"name": "get_detachments_and_abilities", "arguments": {}},
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
