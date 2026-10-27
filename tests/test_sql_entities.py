"""Offline entity-matching and contextual disambiguation contract.

See docs/sql_entity_matching.md for the rules and implementation stages.
"""

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sql_retriever import retrieve_sql
from sql_retriever.entities import resolve_entities, resolved_bindings, validate_entities
from sql_retriever.result import RetrievalFailure
from sql_retriever.safety import ExecutionLimits


class EntityMatchingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "entities.sqlite"
        with closing(sqlite3.connect(self.path)) as connection:
            connection.executescript("""
                CREATE TABLE factions (Id TEXT, name TEXT);
                CREATE TABLE source (Id TEXT, name TEXT);
                CREATE TABLE datasheets (
                    Id TEXT, name TEXT, factionId TEXT, sourceId TEXT,
                    role TEXT DEFAULT '', legend TEXT DEFAULT '', link TEXT DEFAULT ''
                );
                INSERT INTO factions VALUES
                    ('SM', 'Space Marines'), ('CSM', 'Chaos Space Marines'),
                    ('DG', 'Death Guard'), ('ORK', 'Orks');
                INSERT INTO source VALUES
                    ('000000139', 'Space Marines'), ('000000162', 'Black Templars'),
                    ('000000356', 'Space Marines (Warhammer Legends)'),
                    ('000000014', 'Death Guard'), ('000000046', 'Orks'),
                    ('000000377', 'Orks (Warhammer Legends)');
                INSERT INTO datasheets (Id, name, factionId, sourceId) VALUES
                    ('000000148', 'Imperial Space Marine', 'SM', '000000356'),
                    ('000000140', 'Techmarine', 'SM', '000000139'),
                    ('000001044', 'Plague Marines', 'DG', '000000014'),
                    ('000003584', 'Plague Marines', 'CSM', '000000014'),
                    ('000000001', 'Warboss', 'ORK', '000000046'),
                    ('000000002', 'Warboss In Mega Armour', 'ORK', '000000046'),
                    ('000000003', 'Warboss On Warbike', 'ORK', '000000377'),
                    ('000002705', 'Gladiator Lancer', 'SM', '000000139'),
                    ('000002787', 'Gladiator Lancer', 'SM', '000000162');
                CREATE TABLE abilities (Id TEXT, name TEXT, factionId TEXT, description TEXT);
                CREATE TABLE datasheets_abilities (
                    DatasheetId TEXT, line TEXT, abilityId TEXT, name TEXT, description TEXT
                );
                INSERT INTO abilities VALUES
                    ('000008346', 'Leader', '', 'Catalog core rules'),
                    ('catalog-only', 'Catalog Only', 'ORK', 'Catalog-only rules'),
                    ('faction-rule', 'Faction Rule', 'ORK', 'Ork rules'),
                    ('faction-rule', 'Faction Rule', 'DG', 'Death Guard rules');
                INSERT INTO datasheets_abilities VALUES
                    ('000000001', '1', '000008346', '', ''),
                    ('000000001', '3', '', 'Might is Right', 'Warboss card rules'),
                    ('000000002', '3', '', 'Might is Right', 'Mega Armour card rules'),
                    ('000000001', '4', '', 'Da Biggest and da Best', 'Other card rules'),
                    ('000000001', '5', 'faction-rule', 'Card Override', 'Override rules');
            """)
            connection.commit()

    def retrieve_unit(self, value, *, match="name", faction=None, source=None, scoped=False):
        entities = [{"key": "unit", "kind": "unit", "match": match, "value": value}]
        question = f"Get metadata for {value}"
        for kind, context in (("faction", faction), ("source", source)):
            if context is not None:
                question += f"; {kind}: {context}"
                entities.append({"key": kind, "kind": kind, "match": "name", "value": context})
                if scoped:
                    entities[0][f"{kind}_key"] = kind
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(text=json.dumps({
            "mode": "fixed", "function": "get_unit_all_metadata",
            "arguments": {"d_id": "@unit"}, "entities": entities, "message": "",
        }))
        result = retrieve_sql(question, client=client, db_path=self.path, timing=False)
        client.models.generate_content.assert_called_once()
        return result

    def assert_unit(self, result, unit_id, name):
        self.assertEqual(result.status, "ok", result)
        self.assertEqual(result.parameters, {"d_id": unit_id})
        self.assertEqual([row[:2] for row in result.rows], [(unit_id, name)])

    def assert_candidates(self, result, unit_ids):
        self.assertEqual(result.status, "needs_clarification", result)
        self.assertCountEqual([candidate["id"] for candidate in result.candidates], unit_ids)
        self.assertEqual(result.rows, [])
        self.assertIsNone(result.sql)

    def resolve_ability(self, value, *, unit=None, faction=None, source=None, line=None, match="name"):
        ability = {"key": "ability", "kind": "ability", "match": match, "value": value}
        entities = [ability]
        question = f"Get rules for {value}"
        for kind, context in (("unit", unit), ("faction", faction), ("source", source)):
            if context is not None:
                question += f"; {kind}: {context}"
                entities.append({"key": kind, "kind": kind, "match": "name", "value": context})
                ability[f"{kind}_key"] = kind
        if line is not None:
            ability["line"] = line
            question += f"; line: {line}"
        return resolve_entities(validate_entities(entities, question), self.path)

    def test_exact_name_wins_over_similar_names_and_longer_variants(self):
        for name, unit_id in (("Imperial Space Marine", "000000148"), ("Warboss", "000000001")):
            with self.subTest(name=name):
                self.assert_unit(self.retrieve_unit(name), unit_id, name)

    def test_names_ignore_case_and_surrounding_whitespace(self):
        self.assert_unit(self.retrieve_unit("  iMpErIaL sPaCe MaRiNe  "),
                         "000000148", "Imperial Space Marine")

    def test_ids_are_exact_strings_with_leading_zeroes(self):
        self.assert_unit(self.retrieve_unit("000000148", match="id"),
                         "000000148", "Imperial Space Marine")
        for value in ("148", "00000014B"):
            with self.subTest(value=value):
                self.assert_candidates(self.retrieve_unit(value, match="id"), [])

    def test_unknown_name_has_no_invented_candidate(self):
        self.assert_candidates(self.retrieve_unit("Quantum Banana Cruiser"), [])

    def test_same_name_in_different_factions_requires_clarification(self):
        self.assert_candidates(self.retrieve_unit("Plague Marines"), ["000001044", "000003584"])

    def test_explicit_faction_disambiguates(self):
        self.assert_unit(self.retrieve_unit("Plague Marines", faction="Death Guard"),
                         "000001044", "Plague Marines")

    def test_explicit_faction_conflict_does_not_widen_search(self):
        self.assert_candidates(self.retrieve_unit("Imperial Space Marine", faction="Orks"), [])

    def test_same_name_and_faction_still_requires_clarification(self):
        self.assert_candidates(self.retrieve_unit("Gladiator Lancer", faction="Space Marines"),
                               ["000002705", "000002787"])

    def test_resolution_retains_verified_context_when_later_lookup_needs_clarification(self):
        empty = resolve_entities([], self.path)
        self.assertEqual((empty.status, empty.resolved, empty.candidates), ("resolved", {}, []))
        cases = [
            ("Imperial Space Marine", "resolved", {"faction": "SM", "unit": "000000148"}, []),
            ("Gladiator Lancer", "needs_clarification", {"faction": "SM"},
             ["000002705", "000002787"]),
            ("Quantum Banana Cruiser", "needs_clarification", {"faction": "SM"}, []),
        ]
        for name, status, resolved_ids, candidate_ids in cases:
            with self.subTest(name=name):
                resolution = resolve_entities([
                    {"key": "unit", "kind": "unit", "match": "name", "value": name},
                    {"key": "faction", "kind": "faction", "match": "name", "value": "Space Marines"},
                ], self.path)
                self.assertEqual(resolution.status, status)
                self.assertEqual({key: entity["id"] for key, entity in resolution.resolved.items()},
                                 resolved_ids)
                self.assertCountEqual([candidate["id"] for candidate in resolution.candidates],
                                      candidate_ids)
                self.assertEqual(bool(resolution.message), status == "needs_clarification")
                self.assertFalse(resolution.truncated)

    def test_truncated_entity_candidates_remain_clarification_in_public_result(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.executemany(
                "INSERT INTO datasheets (Id, name, factionId, sourceId) VALUES (?, ?, ?, ?)",
                [(f"repeated-{number:03}", "Repeated unit", "SM", "000000139") for number in range(201)],
            )
            connection.commit()
        result = self.retrieve_unit("Repeated unit", faction="Space Marines")
        self.assert_candidates(result, [f"repeated-{number:03}" for number in range(200)])
        self.assertEqual(result.strategy, "fixed")
        self.assertTrue(result.truncated)
        self.assertTrue(result.message)

    def test_internal_whitespace_is_normalized(self):
        self.assert_unit(self.retrieve_unit("Imperial  Space\tMarine"),
                         "000000148", "Imperial Space Marine")

    def test_full_width_names_are_normalized(self):
        self.assert_unit(self.retrieve_unit("Ｉｍｐｅｒｉａｌ　Ｓｐａｃｅ　Ｍａｒｉｎｅ"),
                         "000000148", "Imperial Space Marine")

    def test_unique_complete_phrase_resolves_within_unit_kind(self):
        self.assert_unit(self.retrieve_unit("Space Marine"), "000000148", "Imperial Space Marine")

    def test_partial_name_with_multiple_identities_requires_clarification(self):
        self.assert_candidates(self.retrieve_unit("Plague"), ["000001044", "000003584"])

    def test_misspelling_suggests_the_correct_unit_without_auto_accepting(self):
        result = self.retrieve_unit("impireal spave maeime")
        self.assertEqual(result.status, "needs_clarification", result)
        self.assertGreater(len(result.candidates), 0)
        self.assertLessEqual(len(result.candidates), 5)
        self.assertEqual((result.candidates[0]["id"], result.candidates[0]["name"]),
                         ("000000148", "Imperial Space Marine"))
        self.assertEqual(result.rows, [])
        self.assertIsNone(result.sql)

    def test_short_misspelling_ranks_base_name_before_variants(self):
        result = self.retrieve_unit("Warbos")
        self.assertEqual(result.status, "needs_clarification", result)
        self.assertGreater(len(result.candidates), 0)
        self.assertLessEqual(len(result.candidates), 5)
        self.assertEqual(result.candidates[0]["id"], "000000001")
        self.assertEqual(result.rows, [])
        self.assertIsNone(result.sql)

    def test_name_search_reaches_entities_beyond_business_row_limit(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.executemany(
                "INSERT INTO datasheets (Id, name, factionId, sourceId) VALUES (?, ?, ?, ?)",
                [(f"000000000-{number:03}", f"Reserve entry {number:03}", "SM", "000000139")
                 for number in range(250)],
            )
            connection.commit()
        self.assert_unit(self.retrieve_unit("Space Marine"), "000000148", "Imperial Space Marine")
        result = self.retrieve_unit("impireal spave maeime")
        self.assertEqual(result.status, "needs_clarification", result)
        self.assertGreater(len(result.candidates), 0)
        self.assertEqual(result.candidates[0]["id"], "000000148")

    def test_incomplete_name_catalog_cannot_be_used_to_accept_a_match(self):
        with patch("sql_retriever.entities.NAME_LOOKUP_LIMITS", ExecutionLimits(rows=1000, bytes=100)):
            result = self.retrieve_unit("Warboss")
        self.assert_candidates(result, [])
        self.assertTrue(result.truncated)
        self.assertIn("budget", result.message)

    def test_word_fragment_is_not_automatically_accepted_as_complete_word(self):
        result = self.retrieve_unit("Tech")
        self.assertEqual(result.status, "needs_clarification", result)
        self.assertEqual(result.rows, [])
        self.assertIsNone(result.sql)

    def test_spelling_candidates_respect_explicit_faction(self):
        self.assert_candidates(self.retrieve_unit("Plage Marins", faction="Death Guard"),
                               ["000001044"])

    def test_spelling_candidates_are_bounded_and_keep_distinct_identities(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.executemany(
                "INSERT INTO datasheets (Id, name, factionId, sourceId) VALUES (?, ?, ?, ?)",
                [(f"extra-warboss-{number}", "Warboss", "ORK", "000000046") for number in range(8)],
            )
            connection.commit()
        result = self.retrieve_unit("Warbos")
        self.assertEqual(result.status, "needs_clarification", result)
        self.assertEqual(len(result.candidates), 5)
        self.assertEqual(len({candidate["id"] for candidate in result.candidates}), 5)
        self.assertEqual(result.candidates[0]["id"], "000000001")
        self.assertTrue(result.truncated)
        self.assertEqual(result.rows, [])
        self.assertIsNone(result.sql)

    def test_same_faction_candidates_include_distinguishing_sources(self):
        result = self.retrieve_unit("Gladiator Lancer", faction="Space Marines")
        self.assert_candidates(result, ["000002705", "000002787"])
        self.assertCountEqual(
            [(candidate["id"], candidate["faction_id"], candidate["source_id"],
              candidate["source_name"]) for candidate in result.candidates],
            [("000002705", "SM", "000000139", "Space Marines"),
             ("000002787", "SM", "000000162", "Black Templars")],
        )

    def test_explicit_source_disambiguates_same_faction_name(self):
        for scoped in (False, True):
            with self.subTest(scoped=scoped):
                self.assert_unit(self.retrieve_unit("Gladiator Lancer", faction="Space Marines",
                                                   source="Black Templars", scoped=scoped),
                                 "000002787", "Gladiator Lancer")

    def test_explicit_contexts_intersect_without_widening_name_or_id_search(self):
        cases = [
            ("Gladiator Lancer", "name", "Space Marines", "Orks"),
            ("Gladiator Lancer", "name", "Orks", "Black Templars"),
            ("000002705", "id", "Space Marines", "Black Templars"),
        ]
        for value, match, faction, source in cases:
            with self.subTest(value=value, faction=faction, source=source):
                self.assert_candidates(self.retrieve_unit(value, match=match, faction=faction,
                                                         source=source), [])

    def test_source_context_applies_to_phrase_and_spelling_search(self):
        self.assert_unit(self.retrieve_unit("Gladiator", source="Black Templars"),
                         "000002787", "Gladiator Lancer")
        result = self.retrieve_unit("Gladiatr Lancer", source="Black Templars")
        self.assert_candidates(result, ["000002787"])
        self.assertEqual(result.candidates[0]["source_name"], "Black Templars")
        self.assertEqual(result.candidates[0]["faction_name"], "Space Marines")

    def test_per_entity_references_keep_faction_and_source_contexts_separate(self):
        cases = [
            ("Plague Marines", "faction", "Death Guard", "Chaos Space Marines",
             "000001044", "000003584"),
            ("Gladiator Lancer", "source", "Space Marines", "Black Templars",
             "000002705", "000002787"),
        ]
        for name, kind, first, second, first_id, second_id in cases:
            with self.subTest(kind=kind):
                entities = [
                    {"key": "first_unit", "kind": "unit", "match": "name", "value": name,
                     f"{kind}_key": "first_context"},
                    {"key": "second_unit", "kind": "unit", "match": "name", "value": name,
                     f"{kind}_key": "second_context"},
                    {"key": "first_context", "kind": kind, "match": "name", "value": first},
                    {"key": "second_context", "kind": kind, "match": "name", "value": second},
                ]
                question = f"Compare {name} from {first} with {name} from {second}."
                resolution = resolve_entities(validate_entities(entities, question), self.path)
                self.assertEqual(resolution.status, "resolved", resolution)
                self.assertEqual(resolution.resolved["first_unit"]["id"], first_id)
                self.assertEqual(resolution.resolved["second_unit"]["id"], second_id)

    def test_multiple_unreferenced_contexts_require_clarification(self):
        for kind, first, second in (("faction", "Space Marines", "Orks"),
                                    ("source", "Space Marines", "Black Templars")):
            with self.subTest(kind=kind):
                entities = [
                    {"key": "unit", "kind": "unit", "match": "name", "value": "Imperial Space Marine"},
                    {"key": "first", "kind": kind, "match": "name", "value": first},
                    {"key": "second", "kind": kind, "match": "name", "value": second},
                ]
                resolution = resolve_entities(entities, self.path)
                self.assertEqual(resolution.status, "needs_clarification", resolution)
                self.assertNotIn("unit", resolution.resolved)
                self.assertCountEqual([candidate["name"] for candidate in resolution.candidates],
                                      [first, second])
                self.assertIn("Imperial Space Marine", resolution.message)

    def test_referenced_context_does_not_leak_to_other_entities(self):
        for kind, name, context, unit_id in (
            ("faction", "Imperial Space Marine", "Space Marines", "000000148"),
            ("source", "Gladiator Lancer", "Black Templars", "000002787"),
        ):
            with self.subTest(kind=kind):
                entities = [
                    {"key": "first_unit", "kind": "unit", "match": "name", "value": name,
                     f"{kind}_key": "context"},
                    {"key": "other_unit", "kind": "unit", "match": "name", "value": "Warboss"},
                    {"key": "context", "kind": kind, "match": "name", "value": context},
                ]
                question = f"Compare {name} from {context} with Warboss."
                resolution = resolve_entities(validate_entities(entities, question), self.path)
                self.assertEqual(resolution.status, "resolved", resolution)
                self.assertEqual(resolution.resolved["first_unit"]["id"], unit_id)
                self.assertEqual(resolution.resolved["other_unit"]["id"], "000000001")

    def test_invalid_context_references_are_rejected_at_model_boundary(self):
        cases = [
            {"faction_key": "missing"}, {"faction_key": "source"},
            {"source_key": "faction"}, {"source_key": 123},
            {"kind": "faction", "value": "Space Marines", "faction_key": "faction"},
        ]
        for fields in cases:
            with self.subTest(fields=fields):
                entities = [
                    {"key": "unit", "kind": "unit", "match": "name", "value": "Imperial Space Marine",
                     **fields},
                    {"key": "faction", "kind": "faction", "match": "name", "value": "Space Marines"},
                    {"key": "source", "kind": "source", "match": "name", "value": "Black Templars"},
                ]
                with self.assertRaises(RetrievalFailure) as failure:
                    validate_entities(entities, "Imperial Space Marine, Space Marines, Black Templars")
                self.assertEqual(failure.exception.code, "invalid_model_output")

    def test_matching_evidence_retains_original_text_and_canonical_identity(self):
        cases = [("  iMpErIaL  Space Marine  ", "name", "exact"),
                 ("Space Marine", "name", "phrase"),
                 ("impireal spave maeime", "name", "fuzzy"),
                 ("000000148", "id", "id")]
        for value, match, match_type in cases:
            with self.subTest(value=value):
                resolution = resolve_entities([
                    {"key": "unit", "kind": "unit", "match": match, "value": value},
                ], self.path)
                candidate = (resolution.candidates[0] if match_type == "fuzzy" else
                             resolution.resolved["unit"])
                self.assertEqual(candidate["matched_text"], value)
                self.assertEqual(candidate["match_type"], match_type)
                self.assertEqual(candidate["id"], "000000148")
                self.assertEqual(candidate["name"], "Imperial Space Marine")
                self.assertEqual(candidate["source_name"], "Space Marines (Warhammer Legends)")

    def test_card_only_ability_uses_real_owner_and_line_without_catalog_id(self):
        resolution = self.resolve_ability("Might is Right", unit="Warboss")
        self.assertEqual(resolution.status, "resolved", resolution)
        ability = resolution.resolved["ability"]
        self.assertNotIn("id", ability)
        self.assertNotIn("ability_id", ability)
        self.assertEqual((ability["unit_id"], ability["unit_name"], ability["line"]),
                         ("000000001", "Warboss", "3"))
        self.assertEqual(resolved_bindings(resolution.resolved),
                         {"unit_id": "000000001", "ability_unit_id": "000000001", "ability_line": "3"})

    def test_same_card_ability_name_on_multiple_units_requires_clarification(self):
        resolution = self.resolve_ability("Might is Right")
        self.assertEqual(resolution.status, "needs_clarification", resolution)
        self.assertCountEqual([(candidate["unit_id"], candidate["line"]) for candidate in resolution.candidates],
                              [("000000001", "3"), ("000000002", "3")])
        self.assertTrue(all("id" not in candidate for candidate in resolution.candidates))

    def test_catalog_and_inherited_card_ability_preserve_their_own_identities(self):
        catalog = self.resolve_ability("Leader", faction="Orks")
        self.assertEqual(catalog.status, "resolved", catalog)
        self.assertEqual(catalog.resolved["ability"]["id"], "000008346")
        for value, match in (("Leader", "name"), ("000008346", "id")):
            with self.subTest(match=match):
                card = self.resolve_ability(value, unit="Warboss", match=match)
                self.assertEqual(card.status, "resolved", card)
                ability = card.resolved["ability"]
                self.assertNotIn("id", ability)
                self.assertEqual((ability["ability_id"], ability["unit_id"], ability["line"]),
                                 ("000008346", "000000001", "1"))

    def test_card_ability_search_respects_owner_source_and_spelling_policy(self):
        typo = self.resolve_ability("Migt is Rigt", unit="Warboss")
        self.assertEqual(typo.status, "needs_clarification", typo)
        self.assertEqual((typo.candidates[0]["unit_id"], typo.candidates[0]["line"]), ("000000001", "3"))
        conflict = self.resolve_ability("Might is Right", unit="Warboss", source="Black Templars")
        self.assertEqual(conflict.status, "needs_clarification", conflict)
        self.assertEqual(conflict.candidates, [])

    def test_explicit_card_line_disambiguates_without_becoming_an_ability_id(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("INSERT INTO datasheets_abilities VALUES (?, ?, '', ?, ?)",
                               ("000000001", "6", "Might is Right", "Second card entry"))
            connection.commit()
        ambiguous = self.resolve_ability("Might is Right", unit="Warboss")
        self.assertEqual(ambiguous.status, "needs_clarification", ambiguous)
        self.assertCountEqual([candidate["line"] for candidate in ambiguous.candidates], ["3", "6"])
        selected = self.resolve_ability("Might is Right", unit="Warboss", line="3")
        self.assertEqual(selected.status, "resolved", selected)
        self.assertEqual(selected.resolved["ability"]["line"], "3")
        self.assertNotIn("id", selected.resolved["ability"])
        self.assertEqual(self.resolve_ability("3", unit="Warboss", match="id").candidates, [])
        with self.assertRaises(RetrievalFailure):
            self.resolve_ability("Might is Right", line="3")

    def test_generated_card_query_requires_both_real_identity_bindings(self):
        route = {"mode": "generated", "function": None, "arguments": {}, "message": "", "entities": [
            {"key": "ability", "kind": "ability", "match": "name", "value": "Might is Right",
             "unit_key": "unit"},
            {"key": "unit", "kind": "unit", "match": "name", "value": "Warboss"},
        ]}
        sql = ("SELECT da.name, da.description FROM datasheets_abilities da "
               "JOIN datasheets d ON d.Id=da.DatasheetId "
               "WHERE da.DatasheetId=:ability_unit_id AND da.line=:ability_line AND d.Id=:unit_id")
        required = {"unit_id": "000000001", "ability_unit_id": "000000001", "ability_line": "3"}
        cases = [(sql, required, "ok"), (sql, {**required, "ability_line": "6"}, "error"),
                 (sql, {key: value for key, value in required.items() if key != "ability_line"}, "error"),
                 (sql.replace(" AND da.line=:ability_line", ""), required, "error"),
                 (sql.replace("JOIN datasheets d", "JOIN datasheets_abilities other ON 1=1 JOIN datasheets d")
                  .replace("da.line=:ability_line", "other.line=:ability_line"), required, "error")]
        for generated_sql, parameters, status in cases:
            with self.subTest(sql=generated_sql, parameters=parameters):
                client = Mock()
                client.models.generate_content.side_effect = [SimpleNamespace(text=json.dumps(value)) for value in (
                    route, {"mode": "query", "sql": generated_sql, "parameters": parameters, "message": ""},
                )]
                result = retrieve_sql("Get Might is Right rules for Warboss", client=client,
                                      db_path=self.path, timing=False)
                self.assertEqual(result.status, status, result)
                payload = json.loads(client.models.generate_content.call_args.kwargs["contents"])
                self.assertEqual(payload["required_bindings"], required)
                if status == "ok":
                    self.assertEqual(result.rows, [("Might is Right", "Warboss card rules")])
                else:
                    self.assertEqual(result.error_code, "sql_policy")

    def test_catalog_ability_id_and_faction_must_filter_the_same_catalog_alias(self):
        route = {"mode": "generated", "function": None, "arguments": {}, "message": "", "entities": [
            {"key": "ability", "kind": "ability", "match": "name", "value": "Faction Rule", "faction_key": "faction"},
            {"key": "faction", "kind": "faction", "match": "name", "value": "Orks"},
        ]}
        parameters = {"ability_id": "faction-rule", "ability_faction_id": "ORK", "faction_id": "ORK"}
        sql = ("SELECT a.description FROM abilities a WHERE a.Id=:ability_id "
               "AND a.factionId=:ability_faction_id AND a.factionId=:faction_id")
        cases = [(sql, "ok"),
                 (sql.replace(" AND a.factionId=:ability_faction_id", ""), "error"),
                 (sql.replace("abilities a", "abilities a JOIN abilities b ON b.Id=a.Id")
                  .replace("a.factionId=:ability_faction_id", "b.factionId=:ability_faction_id"), "error")]
        for generated_sql, status in cases:
            with self.subTest(sql=generated_sql):
                client = Mock()
                client.models.generate_content.side_effect = [SimpleNamespace(text=json.dumps(value)) for value in (
                    route, {"mode": "query", "sql": generated_sql, "parameters": parameters, "message": ""})]
                result = retrieve_sql("Get Faction Rule rules for Orks", client=client, db_path=self.path, timing=False)
                self.assertEqual(result.status, status, result)
                if status == "ok":
                    self.assertEqual(result.rows, [("Ork rules",)])
                else:
                    self.assertEqual(result.error_code, "sql_policy")
        resolution = self.resolve_ability("Leader", faction="Orks")
        self.assertEqual(resolved_bindings(resolution.resolved),
                         {"faction_id": "ORK", "ability_id": "000008346", "ability_faction_id": ""})
        route["entities"][0]["value"] = "Leader"
        parameters.update(ability_id="000008346", ability_faction_id="")
        client = Mock()
        client.models.generate_content.side_effect = [SimpleNamespace(text=json.dumps(value)) for value in (
            route, {"mode": "query", "sql": "SELECT a.description FROM abilities a JOIN factions f "
                    "ON f.Id=:faction_id WHERE a.Id=:ability_id AND a.factionId=:ability_faction_id",
                    "parameters": parameters, "message": ""})]
        result = retrieve_sql("Get Leader rules for Orks", client=client, db_path=self.path, timing=False)
        self.assertEqual(result.rows, [("Catalog core rules",)], result)

    def test_model_entity_keys_cannot_overwrite_card_identity_bindings(self):
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(text=json.dumps({
            "mode": "generated", "function": None, "arguments": {}, "message": "", "entities": [
                {"key": "ability", "kind": "ability", "match": "name", "value": "Might is Right", "unit_key": "owner"},
                {"key": "owner", "kind": "unit", "match": "name", "value": "Warboss"},
                {"key": "ability_unit", "kind": "unit", "match": "name", "value": "Warboss In Mega Armour"},
            ]}))
        result = retrieve_sql("Compare Might is Right for Warboss with Warboss In Mega Armour",
                              client=client, db_path=self.path, timing=False)
        self.assertEqual(result.error_code, "invalid_model_output", result)
        client.models.generate_content.assert_called_once()


if __name__ == "__main__":
    unittest.main()
