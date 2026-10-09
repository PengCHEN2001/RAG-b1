"""Offline golden cases for hybrid retrieval and independent SQL safety controls."""

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sql_retriver import RetrievalResult, retrieve_sql
from sql_retriver.entities import resolve_entities, validate_entities
from sql_retriver.result import CompilationFailure, PolicyViolation, RetrievalFailure
from sql_retriver.safety import ExecutionLimits, execute_bounded, install_authorizer, validate_sql
from sql_retriver.schema import inspect_schema, integer_expression, open_readonly


def entity(key="unit", kind="unit", value="Warboss", match="name"):
    return {"key": key, "kind": kind, "match": match, "value": value}


def intent(mode="generated", function=None, arguments=None, entities=None, message=""):
    return {"mode": mode, "function": function, "arguments": arguments or {},
            "entities": entities or [], "message": message}


def query(sql, parameters=None):
    return {"mode": "query", "sql": sql, "parameters": parameters or {}, "message": ""}


def client_for(*responses):
    client = Mock()
    client.models.generate_content.side_effect = [
        SimpleNamespace(text=json.dumps(response)) for response in responses]
    return client


class DatabaseFixture(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "rules.sqlite"
        with closing(sqlite3.connect(self.path)) as c:
            c.executescript('''
                CREATE TABLE factions (Id TEXT, name TEXT);
                CREATE TABLE datasheets (Id TEXT, name TEXT, factionId TEXT,
                    role TEXT, legend TEXT, link TEXT);
                CREATE TABLE abilities (Id TEXT, name TEXT, description TEXT, factionId TEXT);
                CREATE TABLE datasheets_abilities (DatasheetId TEXT, abilityId TEXT, name TEXT, description TEXT);
                CREATE TABLE datasheets_keywords (DatasheetId TEXT, keyword TEXT, model TEXT, isFactionKeyword TEXT);
                CREATE TABLE stratagems (id TEXT, name TEXT, type TEXT, cpCost TEXT, phase TEXT, description TEXT);
                CREATE TABLE datasheets_stratagems (DatasheetId TEXT, stratagemId TEXT);
                CREATE TABLE datasheets_wargear (DatasheetId TEXT, name TEXT, type TEXT,
                    "range" TEXT, a TEXT, bsWs TEXT, s TEXT, ap TEXT, d TEXT);
                CREATE TABLE detachments (Id TEXT, name TEXT, factionId TEXT);
                CREATE TABLE detachment_abilities (Id TEXT, name TEXT, detachmentId TEXT);
                CREATE TABLE datasheets_models (DatasheetId TEXT, name TEXT, t TEXT, sv TEXT);
                CREATE TABLE datasheets_models_cost (DatasheetId TEXT, description TEXT, cost TEXT);
                CREATE TABLE datasheets_leader (LeaderId TEXT, attachedId TEXT);
                CREATE TABLE private_notes (value TEXT);
                INSERT INTO factions VALUES ('ORK', 'Orks'), ('TYR', 'Tyranids');
                INSERT INTO datasheets VALUES
                    ('000000001', 'Warboss', 'ORK', '', '', ''),
                    ('000000002', 'Boyz', 'ORK', '', '', ''),
                    ('000000003', 'Boyz', 'TYR', '', '', ''),
                    ('000000004', 'Gretchin', 'ORK', '', '', '');
                INSERT INTO abilities VALUES ('a1', 'Core', 'Catalog', 'ORK'), ('a1', 'Core', 'Catalog', 'TYR');
                INSERT INTO datasheets_abilities VALUES ('000000001', 'a1', 'Card', 'Card rules');
                INSERT INTO datasheets_keywords VALUES
                    ('000000001', 'INFANTRY', 'Warboss', 'false'),
                    ('000000001', 'INFANTRY', 'Warboss', 'false'),
                    ('000000002', 'INFANTRY', 'Boyz', 'false');
                INSERT INTO stratagems VALUES ('s1', 'Test', 'Battle Tactic', '1', 'Fight', 'Rules');
                INSERT INTO datasheets_stratagems VALUES ('000000001', 's1');
                INSERT INTO datasheets_wargear VALUES ('000000001', 'Axe', 'Melee', 'Melee', 'D6', '2+', '10', '-2', 'D3');
                INSERT INTO detachments VALUES ('dt1', 'War Horde', 'ORK'), ('dt2', 'Missing', 'ORK');
                INSERT INTO detachment_abilities VALUES ('da1', 'Detachment rule', 'dt1');
                INSERT INTO datasheets_models VALUES
                    ('000000001', 'Warboss', '10', '2+'), ('000000002', 'Boyz', '9', '4+'),
                    ('000000003', 'Invalid', 'D6', ''), ('000000004', 'Empty', '', '');
                INSERT INTO datasheets_models_cost VALUES
                    ('000000001', '1 model', '90'), ('000000001', '2 models', '180'),
                    ('000000002', '10 models', '100'), ('000000002', '20 models', '200');
                INSERT INTO datasheets_leader VALUES ('000000001', '000000002');
                INSERT INTO private_notes VALUES ('not approved');
            ''')
            c.commit()
        self.schema = inspect_schema(self.path)

    def run_sql(self, sql, parameters=None):
        parameters = parameters or {}
        safe = validate_sql(sql, parameters, self.schema)
        return execute_bounded(self.path, safe, parameters, self.schema)

    def retrieve(self, question, *responses):
        client = client_for(*responses)
        result = retrieve_sql(question, client=client, db_path=self.path)
        return result, client


class HybridRetrievalTests(DatabaseFixture):
    def test_all_fixed_routes_use_one_model_call(self):
        functions = ["get_unit_all_metadata", "get_unit_abilities", "get_unit_keywords",
                     "get_unit_stratagems", "get_unit_weapons", "get_units_by_faction",
                     "get_detachments_and_abilities"]
        for function in functions:
            with self.subTest(function=function):
                faction = function in functions[-2:]
                entities = [entity("faction", "faction", "Orks")] if faction else [entity()]
                arguments = {"faction_id": "@faction"} if faction else {"d_id": "@unit"}
                result, client = self.retrieve("Get Warboss metadata for Orks", intent(
                    "fixed", function, arguments, entities))
                self.assertEqual(result.status, "ok", result)
                self.assertEqual(result.strategy, "fixed")
                self.assertEqual(client.models.generate_content.call_count, 1)
                self.assertIsNotNone(result.sql)
                config = client.models.generate_content.call_args.kwargs["config"]
                self.assertTrue(config["automatic_function_calling"]["disable"])
                self.assertTrue(result.columns)

    def test_exact_name_and_faction_resolution_preserve_ids(self):
        result, _ = self.retrieve("Get bOyZ metadata for Orks", intent(
            "fixed", "get_unit_all_metadata", {"d_id": "@unit"},
            [entity(value="bOyZ"), entity("faction", "faction", "Orks")]))
        self.assertEqual(result.rows[0][0], "000000002")
        self.assertEqual(result.parameters, {"d_id": "000000002"})

    def test_explicit_id(self):
        result, _ = self.retrieve("Get metadata for 000000001", intent(
            "fixed", "get_unit_all_metadata", {"d_id": "@unit"},
            [entity(value="000000001", match="id")]))
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.rows[0][0], "000000001")

    def test_ambiguous_or_unknown_names_do_not_generate(self):
        for name, count in [("Boyz", 2), ("Warbos", 0)]:
            with self.subTest(name=name):
                result, client = self.retrieve(f"Get {name} metadata", intent(
                    "fixed", "get_unit_all_metadata", {"d_id": "@unit"}, [entity(value=name)]))
                self.assertEqual(result.status, "needs_clarification")
                self.assertEqual(len(result.candidates), count)
                self.assertEqual(client.models.generate_content.call_count, 1)

    def test_empty_fixed_results_never_trigger_fallback(self):
        result, client = self.retrieve("List Orks with role Battleline", intent(
            "fixed", "get_units_by_faction", {"faction_id": "@faction", "role": "Battleline"},
            [entity("faction", "faction", "Orks")]))
        self.assertEqual(result.status, "empty")
        self.assertEqual(client.models.generate_content.call_count, 1)

    def test_generated_keyword_stat_filter_has_no_duplicate_units(self):
        toughness = integer_expression("m.t")
        sql = f'''SELECT d.Id, d.name FROM datasheets d
                  INNER JOIN datasheets_models m ON m.DatasheetId=d.Id
                  WHERE d.factionId=:faction_id AND {toughness}>=:minimum
                  AND EXISTS (SELECT k.DatasheetId FROM datasheets_keywords k
                              WHERE k.DatasheetId=d.Id AND k.keyword=:keyword)
                  ORDER BY d.Id'''
        result, client = self.retrieve("Orks INFANTRY with toughness at least 10", intent(
            entities=[entity("faction", "faction", "Orks")]),
            query(sql, {"faction_id": "ORK", "minimum": 10, "keyword": "INFANTRY"}))
        self.assertEqual(result.status, "ok", result)
        self.assertEqual(result.rows, [("000000001", "Warboss")])
        self.assertEqual(result.strategy, "generated")
        self.assertEqual(client.models.generate_content.call_count, 2)
        content = client.models.generate_content.call_args.kwargs["contents"]
        self.assertNotIn("Card rules", content)

    def test_cost_description_is_not_collapsed_to_universal_cost(self):
        result, _ = self.retrieve("Points for 1 model of Warboss", intent(entities=[entity()]), query(
            "SELECT c.description,c.cost FROM datasheets_models_cost c "
            "WHERE c.DatasheetId=:unit_id AND c.description=:description",
            {"unit_id": "000000001", "description": "1 model"}))
        self.assertEqual(result.rows, [("1 model", "90")])

    def test_leader_relationships(self):
        result, _ = self.retrieve("Which units can Warboss lead?", intent(entities=[entity()]), query(
            "SELECT d.Id,d.name FROM datasheets_leader l INNER JOIN datasheets d "
            "ON d.Id=l.attachedId WHERE l.LeaderId=:unit_id", {"unit_id": "000000001"}))
        self.assertEqual(result.rows, [("000000002", "Boyz")])

    def test_distinct_aggregate(self):
        result, _ = self.retrieve("How many units have INFANTRY?", intent(), query(
            "SELECT COUNT(DISTINCT d.Id) AS total FROM datasheets d "
            "INNER JOIN datasheets_keywords k ON k.DatasheetId=d.Id WHERE k.keyword=:keyword",
            {"keyword": "INFANTRY"}))
        self.assertEqual(result.rows, [(2,)])

    def test_one_schema_repair(self):
        result, client = self.retrieve("List unit names", intent(), query(
            "SELECT imaginary FROM datasheets"), query("SELECT name FROM datasheets ORDER BY Id"))
        self.assertEqual(result.status, "ok", result)
        self.assertEqual(client.models.generate_content.call_count, 3)
        repair = json.loads(client.models.generate_content.call_args.kwargs["contents"])["repair"]
        self.assertEqual(repair["error_code"], "sql_compilation")
        self.assertNotIn("imaginary", json.dumps(repair))
        self.assertNotIn("Card rules", json.dumps(repair))

    def test_second_compilation_failure_stops(self):
        invalid = query("SELECT imaginary FROM datasheets")
        result, client = self.retrieve("List units", intent(), invalid, invalid)
        self.assertEqual(result.error_code, "sql_compilation")
        self.assertEqual(client.models.generate_content.call_count, 3)

    def test_parse_error_can_be_repaired(self):
        result, client = self.retrieve("List units", intent(), query("SELECT ( FROM datasheets"),
                                      query("SELECT Id FROM datasheets"))
        self.assertEqual(result.status, "ok", result)
        self.assertEqual(client.models.generate_content.call_count, 3)

    def test_policy_failure_and_empty_generated_query_never_retry(self):
        cases = [(query("DELETE FROM datasheets"), "error"),
                 (query("SELECT Id FROM datasheets WHERE name=:name", {"name": "Nobody"}), "empty")]
        for response, expected in cases:
            with self.subTest(expected=expected):
                result, client = self.retrieve("Find units", intent(), response)
                self.assertEqual(result.status, expected)
                self.assertEqual(client.models.generate_content.call_count, 2)

    def test_execution_failure_does_not_trigger_repair(self):
        client = client_for(intent(), query("SELECT Id FROM datasheets"))
        from sql_retriver.safety import ExecutedQuery
        with patch("sql_retriver.hybrid.execute_bounded", side_effect=[
                ExecutedQuery([], [], False), RetrievalFailure("query_budget", "Budget exceeded.")]):
            result = retrieve_sql("List units", client=client, db_path=self.path)
        self.assertEqual(result.error_code, "query_budget")
        self.assertEqual(client.models.generate_content.call_count, 2)

    def test_invalid_intent_contract_and_inferred_role(self):
        invalid = [intent("generated", "get_unit_keywords"),
                   intent("fixed", "invented_function"),
                   intent("fixed", "get_unit_all_metadata", {"d_id": "@faction"},
                          [entity("faction", "faction", "Orks")]),
                   intent("fixed", "get_units_by_faction",
                          {"faction_id": "@faction", "role": "Battleline"},
                          [entity("faction", "faction", "Orks")])]
        for response in invalid:
            with self.subTest(response=response):
                result, _ = self.retrieve("List Orks units", response)
                self.assertEqual(result.error_code, "invalid_model_output")

    def test_explicit_outcomes(self):
        for mode in ("unsupported", "needs_clarification"):
            result, client = self.retrieve("Best unit?", intent(mode, message="Define best."))
            if mode == "needs_clarification":
                # Routing cannot claim entity ambiguity before application-owned lookup.
                self.assertEqual(result.error_code, "invalid_model_output")
            else:
                self.assertEqual(result.status, mode)
            self.assertEqual(client.models.generate_content.call_count, 1)
            result, client = self.retrieve("Best unit?", intent(), {
                "mode": mode, "sql": None, "parameters": {}, "message": "Define best."})
            self.assertEqual(result.status, mode)
            self.assertEqual(client.models.generate_content.call_count, 2)

    def test_fabricated_ids_are_not_accepted(self):
        result, _ = self.retrieve("Get Warboss metadata", intent(
            "fixed", "get_unit_all_metadata", {"d_id": "000000001"}, [entity()]))
        self.assertEqual(result.error_code, "invalid_model_output")
        result, _ = self.retrieve("Get Warboss metadata", intent(
            "fixed", "get_unit_all_metadata", {"d_id": "@unit"},
            [entity(value="000000001", match="id")]))
        self.assertEqual(result.error_code, "invalid_model_output")
        result, _ = self.retrieve("Warboss model stats", intent(entities=[entity()]), query(
            "SELECT name FROM datasheets_models WHERE DatasheetId=:unit_id", {"unit_id": "000000002"}))
        self.assertEqual(result.error_code, "sql_policy")

    def test_malformed_model_output_and_api_failures(self):
        for text in (None, "", "not json", "[]", '{}', '{"mode":"fixed","mode":"generated"}'):
            client = Mock()
            client.models.generate_content.return_value = SimpleNamespace(text=text)
            result = retrieve_sql("Get units", client=client, db_path=self.path)
            self.assertEqual(result.error_code, "invalid_model_output")
        client = Mock()
        client.models.generate_content.side_effect = RuntimeError("private provider details")
        result = retrieve_sql("Get units", client=client, db_path=self.path)
        self.assertEqual(result.error_code, "model_error")
        self.assertNotIn("private", result.message)

    def test_invalid_unicode_returns_structured_errors(self):
        result = retrieve_sql("\ud800", client=Mock(), db_path=self.path)
        self.assertEqual(result.error_code, "invalid_question")
        responses = [query("SELECT Id FROM datasheets WHERE name=:name", {"name": "\ud800"}),
                     query("SELECT \ud800 FROM datasheets")]
        for response in responses:
            with self.subTest(response=response):
                result, client = self.retrieve("Find units", intent(), response)
                self.assertEqual(result.error_code, "sql_policy")
                self.assertEqual(client.models.generate_content.call_count, 2)

    def test_injection_strings_are_values(self):
        value = "Warboss'; DROP TABLE datasheets; --"
        result, _ = self.retrieve("Find this name", intent(), query(
            "SELECT Id FROM datasheets WHERE name=:name", {"name": value}))
        self.assertEqual(result.status, "empty")
        self.assertEqual(len(self.run_sql("SELECT Id FROM datasheets").rows), 4)

    def test_root_limit_preserves_order_and_user_limit(self):
        with closing(sqlite3.connect(self.path)) as c:
            c.executemany("INSERT INTO datasheets VALUES (?, ?, 'ORK', '', '', '')",
                          [(f"extra-{n:03}", str(n)) for n in range(250)])
            c.commit()
        result, _ = self.retrieve("List units", intent(), query("SELECT Id FROM datasheets ORDER BY Id DESC"))
        self.assertTrue(result.truncated)
        self.assertEqual(len(result.rows), 200)
        self.assertEqual(result.rows[0], ("extra-249",))
        result, _ = self.retrieve("First 3 units", intent(), query("SELECT Id FROM datasheets ORDER BY Id LIMIT 3"))
        self.assertFalse(result.truncated)
        self.assertEqual(len(result.rows), 3)

    def test_missing_database_invalid_question_and_missing_credentials(self):
        self.assertEqual(retrieve_sql(" ", client=Mock(), db_path=self.path).error_code, "invalid_question")
        self.assertEqual(retrieve_sql("List units", client=Mock(), db_path=self.path.parent / "missing").error_code,
                         "database_error")
        with patch.dict("sys.modules", {"llm.llm": None}):
            result = retrieve_sql("List units", db_path=self.path)
        self.assertEqual(result.error_code, "client_unavailable")


class EntityResolutionTests(DatabaseFixture):
    def test_whitespace_and_case(self):
        entries = validate_entities([entity(value=" warboss ")], "warboss")
        resolved = resolve_entities(entries, self.path, self.schema)
        self.assertEqual(resolved["unit"]["id"], "000000001")

    def test_duplicate_keys_wrong_types_and_substrings_rejected(self):
        for entries in ([entity(), entity()], [None], [entity(value="War")],
                        [dict(entity(), match="fuzzy")], [dict(entity(), value=1)]):
            with self.subTest(entries=entries), self.assertRaises(RetrievalFailure):
                validate_entities(entries, "Warboss")

    def test_other_singular_entities_are_resolved_or_clarified(self):
        resolved = resolve_entities([entity("detachment", "detachment", "War Horde")], self.path, self.schema)
        self.assertEqual(resolved["detachment"]["id"], "dt1")
        result = resolve_entities([entity("ability", "ability", "Core")], self.path, self.schema)
        self.assertEqual(result.status, "needs_clarification")
        self.assertEqual(len(result.candidates), 2)


class SQLSafetyTests(DatabaseFixture):
    def test_grounding_excludes_unexpected_tables(self):
        self.assertNotIn("private_notes", self.schema.tables)
        self.assertNotIn("private_notes", self.schema.prompt())
        self.assertIn("TEXT", self.schema.prompt())
        self.assertIn("cost description", self.schema.prompt())

    def test_ctes_derived_tables_and_correlated_exists(self):
        sql = '''WITH units AS (SELECT d.Id,d.name FROM datasheets d)
            SELECT u.name FROM units u INNER JOIN datasheets_keywords k ON u.Id=k.DatasheetId
            WHERE k.keyword=:keyword ORDER BY u.name'''
        rows = self.run_sql(sql, {"keyword": "INFANTRY"}).rows
        self.assertEqual(len(rows), 3)
        rows = self.run_sql('''SELECT d.name FROM datasheets d WHERE EXISTS
            (SELECT k.DatasheetId FROM datasheets_keywords k WHERE k.DatasheetId=d.Id
            AND k.keyword=:keyword) ORDER BY d.name''', {"keyword": "INFANTRY"}).rows
        self.assertEqual(rows, [("Boyz",), ("Warboss",)])

    def test_integer_guard_handles_plain_signed_integers_only(self):
        values = ["9", "10", "", "D6", "2+", "-2", "-1-", "1.5", " 12 ", "99999999999999999999999"]
        with closing(sqlite3.connect(self.path)) as c:
            c.execute("DELETE FROM datasheets_models")
            c.executemany("INSERT INTO datasheets_models VALUES ('id', ?, ?, '')", [(v, v) for v in values])
            c.commit()
        rows = self.run_sql(f"SELECT m.name, {integer_expression('m.t')} AS value FROM datasheets_models m").rows
        self.assertEqual([r[1] for r in rows], [9, 10, None, None, None, -2, None, None, 12, None])
        numeric = integer_expression("m.t")
        rows = self.run_sql(f"SELECT m.name FROM datasheets_models m WHERE {numeric}>:threshold "
                            f"ORDER BY {numeric}", {"threshold": 9}).rows
        self.assertEqual(rows, [("10",), (" 12 ",)])

    def test_numeric_aggregates_over_guarded_cte(self):
        expression = integer_expression("c.cost")
        rows = self.run_sql(f"WITH costs AS (SELECT {expression} AS n FROM datasheets_models_cost c) "
                            "SELECT SUM(n), AVG(n) FROM costs").rows
        self.assertEqual(rows, [(570, 142.5)])

    def test_raw_numeric_interpretations_are_rejected(self):
        queries = ["SELECT CAST(t AS INTEGER) FROM datasheets_models",
                   "SELECT t FROM datasheets_models WHERE t>:threshold",
                   "SELECT t FROM datasheets_models ORDER BY t",
                   "SELECT t FROM datasheets_models ORDER BY COALESCE(t,:value)",
                   "SELECT t FROM datasheets_models WHERE t BETWEEN :threshold AND :threshold",
                   "SELECT t FROM datasheets_models WHERE t=:threshold",
                   "SELECT SUM(cost) FROM datasheets_models_cost",
                   "SELECT MIN(cost) FROM datasheets_models_cost",
                   "SELECT t + :value FROM datasheets_models"]
        for sql in queries:
            params = {"threshold": 9} if ":threshold" in sql else {"value": 1} if ":value" in sql else {}
            with self.subTest(sql=sql), self.assertRaises(PolicyViolation):
                validate_sql(sql, params, self.schema)

    def test_ast_policy_rejects_unsafe_and_unsupported_constructs(self):
        queries = [
            "DELETE FROM datasheets", "SELECT Id FROM datasheets; DELETE FROM datasheets",
            "PRAGMA table_info(datasheets)", "ATTACH DATABASE :path AS evil",
            "SELECT * FROM datasheets", "SELECT d.* FROM datasheets d",
            "SELECT load_extension(:path) FROM datasheets", "SELECT readfile(:path) FROM datasheets",
            "SELECT name FROM sqlite_master", "SELECT value FROM private_notes",
            "SELECT Id FROM imaginary", "SELECT Id FROM main.datasheets",
            "SELECT Id FROM datasheets UNION SELECT Id FROM datasheets",
            "SELECT ROW_NUMBER() OVER () FROM datasheets", "SELECT randomblob(:size) FROM datasheets",
            "SELECT a.Id FROM datasheets a CROSS JOIN factions f",
            "SELECT a.Id FROM datasheets a, factions f",
            "SELECT a.Id FROM datasheets a INNER JOIN factions f ON a.name=f.name",
            "SELECT a.Id FROM datasheets a JOIN factions f ON a.factionId=f.Id OR a.name=f.name",
            "SELECT d.Id FROM datasheets d WHERE EXISTS (SELECT f.Id FROM factions f WHERE f.name=d.name)",
            "SELECT Id FROM datasheets WHERE name='Warboss'", "SELECT Id FROM datasheets LIMIT :limit",
            "SELECT Id FROM datasheets LIMIT -1", "SELECT Id FROM datasheets LIMIT 1000001",
            "WITH RECURSIVE x AS (SELECT Id FROM datasheets) SELECT Id FROM x",
            "SELECT name FROM pragma_table_info(:table)",
            "SELECT name FROM json_each(:value)", "SELECT Id FROM datasheets WHERE Id=?",
            "SELECT Id FROM datasheets WHERE Id=@id",
        ]
        for sql in queries:
            with self.subTest(sql=sql), self.assertRaises((PolicyViolation, CompilationFailure)):
                validate_sql(sql, {}, self.schema)

    def test_join_and_depth_limits(self):
        sql = "SELECT d.Id FROM datasheets d " + " ".join(
            f"JOIN datasheets_keywords k{n} ON k{n}.DatasheetId=d.Id" for n in range(5))
        with self.assertRaises(PolicyViolation):
            validate_sql(sql, {}, self.schema)
        with self.assertRaises(PolicyViolation):
            validate_sql("SELECT Id FROM (SELECT Id FROM (SELECT Id FROM (SELECT Id FROM datasheets)))", {}, self.schema)
        with self.assertRaises(PolicyViolation):
            validate_sql("SELECT Id FROM datasheets " + " " * (16 * 1024), {}, self.schema)

    def test_parameters_and_ambiguous_columns(self):
        cases = [("SELECT Id FROM datasheets WHERE Id=:id", {}),
                 ("SELECT Id FROM datasheets", {"extra": "x"}),
                 ("SELECT Id FROM datasheets WHERE Id=:id", {"id": []}),
                 ("SELECT Id FROM datasheets WHERE Id=:id", {"id": float("nan")})]
        for sql, params in cases:
            with self.subTest(sql=sql, params=params), self.assertRaises(PolicyViolation):
                validate_sql(sql, params, self.schema)
        with self.assertRaises(CompilationFailure):
            validate_sql("SELECT name FROM datasheets d JOIN factions f ON d.factionId=f.Id", {}, self.schema)

    def test_authorizer_is_independent_of_validator(self):
        unsafe = ["DELETE FROM datasheets", "CREATE TABLE malicious (x)", "PRAGMA user_version",
                  "ATTACH DATABASE ':memory:' AS evil", "SELECT name FROM sqlite_master",
                  "SELECT value FROM private_notes", "SELECT load_extension('evil')",
                  "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x WHERE n<5) SELECT n FROM x"]
        with closing(open_readonly(self.path)) as c:
            install_authorizer(c, self.schema)
            for sql in unsafe:
                with self.subTest(sql=sql), self.assertRaises(sqlite3.DatabaseError):
                    c.execute(sql).fetchall()
            self.assertEqual(c.execute("SELECT COUNT(*) FROM datasheets").fetchone(), (4,))
        self.assertEqual(len(self.run_sql("SELECT Id FROM datasheets").rows), 4)

    def test_execution_budget_row_bytes_and_cleanup(self):
        bounded = execute_bounded(self.path, "SELECT Id FROM datasheets ORDER BY Id", {}, self.schema,
                                  limits=ExecutionLimits(rows=2))
        self.assertEqual(len(bounded.rows), 2)
        self.assertTrue(bounded.truncated)
        bounded = execute_bounded(self.path, "SELECT Id FROM datasheets", {}, self.schema,
                                  limits=ExecutionLimits(bytes=25))
        self.assertTrue(bounded.truncated)
        self.assertLess(len(bounded.rows), 4)
        with self.assertRaises(RetrievalFailure) as error:
            execute_bounded(self.path, "SELECT COUNT(*) FROM datasheets a, datasheets b, datasheets c",
                            {}, self.schema, limits=ExecutionLimits(vm_steps=10))
        self.assertEqual(error.exception.code, "query_budget")
        with self.assertRaises(RetrievalFailure) as error:
            execute_bounded(self.path, "SELECT Id FROM datasheets", {}, self.schema,
                            limits=ExecutionLimits(seconds=0))
        self.assertEqual(error.exception.code, "query_budget")
        # Exclusive access is possible after both success and failure: no leaked connection.
        with closing(sqlite3.connect(self.path, timeout=0.01)) as c:
            c.execute("BEGIN EXCLUSIVE")
            c.rollback()

    def test_compile_only_and_value_budget(self):
        result = execute_bounded(self.path, "SELECT name FROM datasheets", {}, self.schema, compile_only=True)
        self.assertEqual(result.rows, [])
        with closing(sqlite3.connect(self.path)) as c:
            c.execute("UPDATE datasheets SET name=? WHERE Id=?", ("x" * 3000, "000000001"))
            c.commit()
        with self.assertRaises(RetrievalFailure) as error:
            execute_bounded(self.path, "SELECT name FROM datasheets", {}, self.schema,
                            limits=ExecutionLimits(bytes=1024))
        self.assertEqual(error.exception.code, "result_budget")


if __name__ == "__main__":
    unittest.main()
