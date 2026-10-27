"""Essential offline checks for hybrid routing, name resolution, and SQL safety."""

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sql_retriever import retrieve_sql
from sql_retriever.hybrid import DATA_DICTIONARY_PATH, INTEGER_TEMPLATE
from sql_retriever.result import PolicyViolation, RetrievalFailure
from sql_retriever.safety import (
    ExecutionLimits, execute_bounded, install_authorizer, open_readonly,
)


def entity(key="unit", kind="unit", value="Warboss", match="name"):
    return {"key": key, "kind": kind, "match": match, "value": value}


def intent(mode="generated", function=None, arguments=None, entities=None):
    return {"mode": mode, "function": function, "arguments": arguments or {},
            "entities": entities or [], "message": ""}


def query(sql, parameters=None):
    return {"mode": "query", "sql": sql, "parameters": parameters or {}, "message": ""}


def client_for(*responses):
    client = Mock()
    client.models.generate_content.side_effect = [
        SimpleNamespace(text=json.dumps(response)) for response in responses]
    return client


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "rules.sqlite"
        with closing(sqlite3.connect(self.path)) as connection:
            connection.executescript('''
                CREATE TABLE factions (Id TEXT, name TEXT);
                CREATE TABLE source (Id TEXT, name TEXT);
                CREATE TABLE datasheets (Id TEXT, name TEXT, factionId TEXT,
                    role TEXT, legend TEXT, link TEXT, sourceId TEXT DEFAULT '');
                CREATE TABLE datasheets_keywords (DatasheetId TEXT, keyword TEXT);
                CREATE TABLE datasheets_models (
                    DatasheetId TEXT, name TEXT, t TEXT, line TEXT DEFAULT '', m TEXT DEFAULT '',
                    sv TEXT DEFAULT '', invSv TEXT DEFAULT '', invSvDescr TEXT DEFAULT '', w TEXT DEFAULT '',
                    ld TEXT DEFAULT '', oc TEXT DEFAULT ''
                );
                CREATE TABLE private_notes (value TEXT);
                INSERT INTO factions VALUES ('ORK', 'Orks'), ('TYR', 'Tyranids');
                INSERT INTO datasheets (Id, name, factionId, role, legend, link) VALUES
                    ('000000001', 'Warboss', 'ORK', '', '', ''),
                    ('000000002', 'Boyz', 'ORK', '', '', ''),
                    ('000000003', 'Boyz', 'TYR', '', '', ''),
                    ('000000004', 'Gretchin', 'ORK', '', '', '');
                INSERT INTO datasheets_keywords VALUES
                    ('000000001', 'INFANTRY'), ('000000001', 'INFANTRY'),
                    ('000000002', 'INFANTRY');
                INSERT INTO datasheets_models (DatasheetId, name, t) VALUES
                    ('000000001', 'Warboss', '10'), ('000000002', 'Boyz', '9'),
                    ('000000003', 'Dice', 'D6'), ('000000004', 'Empty', ''),
                    ('000000001', 'Signed', '-2'), ('000000002', 'Save', '2+');
            ''')
            connection.commit()

    def retrieve(self, question, *responses):
        client = client_for(*responses)
        return retrieve_sql(question, client=client, db_path=self.path, timing=False), client

    def test_fixed_route_resolves_names_ids_and_faction_context(self):
        cases = [
            ("Warboss", [entity()], "000000001"),
            ("000000001", [entity(value="000000001", match="id")], "000000001"),
            ("bOyZ for Orks", [entity(value="bOyZ"), entity("faction", "faction", "Orks")], "000000002"),
        ]
        for name, entities, expected_id in cases:
            with self.subTest(name=name):
                result, client = self.retrieve(f"Get metadata for {name}", intent(
                    "fixed", "get_unit_all_metadata", {"d_id": "@unit"}, entities))
                self.assertEqual((result.status, result.strategy), ("ok", "fixed"))
                self.assertEqual(result.rows[0][0], expected_id)
                self.assertEqual(result.parameters, {"d_id": expected_id})
                self.assertEqual(client.models.generate_content.call_count, 1)

    def test_ambiguous_or_unknown_names_stop_before_generation(self):
        for name, count in [("Boyz", 2), ("Quantum Banana Cruiser", 0)]:
            with self.subTest(name=name):
                result, client = self.retrieve(f"Get {name} metadata", intent(
                    "fixed", "get_unit_all_metadata", {"d_id": "@unit"}, [entity(value=name)]))
                self.assertEqual(result.status, "needs_clarification")
                self.assertEqual(len(result.candidates), count)
                self.assertEqual(client.models.generate_content.call_count, 1)

    def test_fixed_model_stats_return_all_profiles_and_preserve_symbolic_values(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute('''UPDATE datasheets_models SET m='6"', sv='3+', invSv='5+', invSvDescr='Against ranged attacks',
                w='4', ld='6+', oc='1', line=CASE name WHEN 'Warboss' THEN '1' ELSE '2' END
                WHERE DatasheetId='000000001' ''')
            connection.commit()
        result, client = self.retrieve("What is the Toughness of Warboss?", intent(
            "fixed", "get_unit_model_stats", {"d_id": "@unit"}, [entity()]))
        self.assertEqual((result.status, result.strategy), ("ok", "fixed"), result)
        self.assertEqual(result.parameters, {"d_id": "000000001"})
        profiles = [dict(zip(result.columns, row)) for row in result.rows]
        self.assertEqual([(profile["model_name"], profile["toughness"]) for profile in profiles],
                         [("Warboss", "10"), ("Signed", "-2")])
        for profile in profiles:
            self.assertEqual((profile["movement"], profile["save"], profile["invulnerable_save"],
                              profile["leadership"]), ('6"', '3+', '5+', '6+'))
            self.assertEqual(profile["invulnerable_save_description"], 'Against ranged attacks')
        client.models.generate_content.assert_called_once()

    def test_generated_route_executes_grounded_numeric_query(self):
        numeric_toughness = INTEGER_TEMPLATE.format(column="m.t")
        sql = f'''SELECT d.Id, d.name FROM datasheets d
            INNER JOIN datasheets_models m ON m.DatasheetId=d.Id
            WHERE d.factionId=:faction_id AND {numeric_toughness}>=:minimum
            AND EXISTS (SELECT k.DatasheetId FROM datasheets_keywords k
                        WHERE k.DatasheetId=d.Id AND k.keyword=:keyword)'''
        result, client = self.retrieve("Orks INFANTRY with toughness at least 10", intent(
            entities=[entity("faction", "faction", "Orks")]),
            query(sql, {"faction_id": "ORK", "minimum": 10, "keyword": "INFANTRY"}))
        self.assertEqual((result.status, result.strategy), ("ok", "generated"), result)
        self.assertEqual(result.rows, [("000000001", "Warboss")])
        self.assertEqual(client.models.generate_content.call_count, 2)

    def test_full_dictionary_is_in_both_prompts_and_sql_executes_once(self):
        sql = "SELECT name FROM datasheets ORDER BY Id LIMIT 2"
        with patch("sql_retriever.hybrid.execute_bounded", wraps=execute_bounded) as execute:
            result, client = self.retrieve("List two unit names", intent(), query(sql))
        self.assertEqual(result.rows, [("Warboss",), ("Boyz",)])
        self.assertEqual(result.sql, sql)
        execute.assert_called_once_with(self.path, sql, {})
        dictionary = DATA_DICTIONARY_PATH.read_text(encoding="utf-8")
        for call in client.models.generate_content.call_args_list:
            prompt = call.kwargs["config"]["system_instruction"]
            reference = prompt.split("<data_dictionary>\n", 1)[1].rsplit("\n</data_dictionary>", 1)[0]
            self.assertEqual(reference, dictionary)

    def test_compilation_repair_is_limited_to_one_attempt(self):
        parameters = {"unit_id": "000000001"}
        invalid = query("SELECT imaginary FROM datasheets WHERE Id=:unit_id", parameters)
        valid = query("SELECT name FROM datasheets WHERE Id=:unit_id", parameters)
        for correction, expected in [(valid, "ok"), (invalid, "error")]:
            with self.subTest(expected=expected):
                result, client = self.retrieve("Warboss name", intent(entities=[entity()]), invalid, correction)
                self.assertEqual(result.status, expected)
                self.assertEqual(client.models.generate_content.call_count, 3)
                if expected == "error":
                    self.assertEqual(result.error_code, "sql_compilation")
                repair = json.loads(client.models.generate_content.call_args.kwargs["contents"])["repair"]
                self.assertIn("imaginary", repair["message"])
                self.assertEqual(repair["error_code"], "sql_compilation")
                self.assertEqual(repair["sql"], invalid["sql"])
                self.assertEqual(repair["parameters"], parameters)
                self.assertEqual(set(repair), {"error_code", "message", "sql", "parameters"})
                payload = json.loads(client.models.generate_content.call_args.kwargs["contents"])
                self.assertEqual(payload["required_bindings"], parameters)

        # Parser failures get the same single repair, and a repaired policy violation still stops.
        malformed = query("SELECT name FROM datasheets WHERE (Id=:unit_id", parameters)
        bypass = query("SELECT name FROM datasheets WHERE Id=:unit_id OR 1=1", parameters)
        for corrected, expected in [(valid, None), (bypass, "sql_policy")]:
            result, client = self.retrieve("Warboss name", intent(entities=[entity()]), malformed, corrected)
            self.assertEqual(result.error_code, expected)
            if expected is None:
                self.assertEqual(result.rows, [("Warboss",)])
            self.assertEqual(client.models.generate_content.call_count, 3)
            repair = json.loads(client.models.generate_content.call_args.kwargs["contents"])["repair"]
            self.assertEqual((repair["sql"], repair["parameters"]), (malformed["sql"], parameters))

    def test_empty_policy_and_budget_failures_do_not_retry(self):
        cases = [
            (query("SELECT Id FROM datasheets WHERE name=:name", {"name": "Nobody"}), "empty", None),
            (query("DELETE FROM datasheets"), "error", "sql_policy"),
            (query("SELECT Id FROM datasheets WHERE name=:name"), "error", "sql_policy"),
            (query("SELECT Id FROM datasheets; SELECT Id FROM datasheets"), "error", "sql_policy"),
            (query("SELECT value FROM private_notes"), "error", "sql_policy"),
        ]
        for response, status, code in cases:
            with self.subTest(status=status):
                result, client = self.retrieve("Find units", intent(), response)
                self.assertEqual((result.status, result.error_code), (status, code))
                self.assertEqual(client.models.generate_content.call_count, 2)
        client = client_for(intent(), query("SELECT Id FROM datasheets"))
        with patch("sql_retriever.hybrid.execute_bounded",
                   side_effect=RetrievalFailure("query_budget", "Budget exceeded.")):
            result = retrieve_sql("List units", client=client, db_path=self.path)
        self.assertEqual(result.error_code, "query_budget")
        self.assertEqual(client.models.generate_content.call_count, 2)
        result, client = self.retrieve("List Orks with role Battleline", intent(
            "fixed", "get_units_by_faction", {"faction_id": "@faction", "role": "Battleline"},
            [entity("faction", "faction", "Orks")]))
        self.assertEqual(result.status, "empty")
        self.assertEqual(client.models.generate_content.call_count, 1)

    def test_model_failures_and_premature_clarification_are_explicit(self):
        for text, code in [("not json", "invalid_model_output"),
                           (json.dumps(intent("needs_clarification")), "invalid_model_output")]:
            with self.subTest(text=text):
                client = Mock()
                client.models.generate_content.return_value = SimpleNamespace(text=text)
                result = retrieve_sql("Get Warboss metadata", client=client, db_path=self.path)
                self.assertEqual(result.error_code, code)
                self.assertEqual(client.models.generate_content.call_count, 1)
        client = Mock()
        client.models.generate_content.side_effect = RuntimeError("private provider details")
        result = retrieve_sql("Get units", client=client, db_path=self.path, timing=False)
        self.assertEqual(result.error_code, "model_error")
        self.assertNotIn("private", result.message)

    def test_bindings_preserve_values_and_resolved_ids(self):
        injection = "Warboss'; DROP TABLE datasheets; --"
        result, _ = self.retrieve("Find this name", intent(), query(
            "SELECT Id FROM datasheets WHERE name=:name", {"name": injection}))
        self.assertEqual(result.status, "empty")
        with closing(open_readonly(self.path)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM datasheets").fetchone(), (4,))
        with self.assertRaises(PolicyViolation):
            execute_bounded(self.path, "SELECT Id FROM datasheets WHERE Id=:id", {})
        result, _ = self.retrieve("Warboss model stats", intent(entities=[entity()]), query(
            "SELECT name FROM datasheets_models WHERE DatasheetId=:unit_id", {"unit_id": "000000002"}))
        self.assertEqual(result.error_code, "sql_policy")

    def test_identity_mentions_cannot_bypass_effective_filters(self):
        queries = [
            "SELECT name FROM datasheets WHERE Id='000000002'",
            "SELECT :unit_id, name FROM datasheets WHERE Id='000000002'",
            "SELECT ':unit_id', name FROM datasheets WHERE Id='000000002'",
            "SELECT name FROM datasheets WHERE Id='000000002' -- Id=:unit_id",
            "SELECT name FROM datasheets WHERE name=:unit_id",
            "SELECT name FROM factions WHERE Id=:unit_id",
            "SELECT name FROM datasheets WHERE :unit_id=:unit_id",
            "SELECT name FROM datasheets WHERE Id<>:unit_id",
            "SELECT name FROM datasheets WHERE Id=:unit_id OR 1=1",
            "SELECT name FROM datasheets WHERE NOT (Id=:unit_id)",
            "SELECT CASE WHEN Id=:unit_id THEN name END FROM datasheets",
            "SELECT name FROM datasheets WHERE Id IN (:unit_id, :other)",
            "SELECT name FROM datasheets WHERE Id IN (:unit_id, '000000002')",
            "WITH unused AS (SELECT name FROM datasheets WHERE Id=:unit_id) "
            "SELECT name FROM datasheets WHERE Id='000000002'",
            "SELECT d.name FROM datasheets d WHERE EXISTS "
            "(SELECT x.Id FROM datasheets x WHERE x.Id=:unit_id)",
            "SELECT (SELECT name FROM datasheets WHERE Id=:unit_id), name FROM datasheets",
            "SELECT d.name FROM datasheets d LEFT JOIN factions f "
            "ON f.Id=d.factionId AND d.Id=:unit_id",
            "WITH chosen AS (SELECT Id FROM datasheets WHERE Id=:unit_id) "
            "SELECT d.name FROM datasheets d LEFT JOIN chosen c ON c.Id=d.Id",
            "SELECT name FROM datasheets WHERE Id=:unit_id UNION ALL "
            "SELECT name FROM datasheets WHERE Id='000000002'",
            "WITH datasheets AS (SELECT '000000001' AS Id, 'Wrong' AS name) "
            "SELECT name FROM datasheets WHERE Id=:unit_id",
        ]
        for sql in queries:
            with self.subTest(sql=sql), patch("sql_retriever.hybrid.execute_bounded") as execute:
                result, client = self.retrieve("Warboss name", intent(entities=[entity()]),
                                               query(sql, {"unit_id": "000000001", "other": "000000002"}))
                self.assertEqual(result.error_code, "sql_policy", result)
                self.assertEqual(client.models.generate_content.call_count, 2)
                execute.assert_not_called()

    def test_effective_identity_filters_support_joins_ctes_and_resolved_lists(self):
        queries = [
            "SELECT name FROM datasheets WHERE :unit_id=Id",
            'SELECT D.name FROM datasheets D WHERE D."ID"=:unit_id',
            "SELECT d.name FROM datasheets d JOIN factions f "
            "ON f.Id=d.factionId AND d.Id=:unit_id",
            "SELECT d.name FROM datasheets d LEFT JOIN factions f "
            "ON f.Id=d.factionId WHERE d.Id=:unit_id",
            "SELECT name FROM datasheets WHERE "
            "(Id=:unit_id AND name='Warboss') OR (Id=:unit_id AND name='Other')",
            "WITH chosen AS (SELECT name FROM datasheets WHERE Id=:unit_id) SELECT name FROM chosen",
            "SELECT name FROM (SELECT name FROM datasheets WHERE Id=:unit_id) chosen",
            "SELECT name FROM datasheets WHERE Id=:unit_id UNION "
            "SELECT name FROM datasheets WHERE Id=:unit_id",
        ]
        for sql in queries:
            with self.subTest(sql=sql):
                result, _ = self.retrieve("Warboss name", intent(entities=[entity()]),
                                          query(sql, {"unit_id": "000000001"}))
                self.assertEqual(result.status, "ok", result)
                self.assertEqual(result.rows, [("Warboss",)])
        result, _ = self.retrieve("Warboss and Gretchin names", intent(entities=[
            entity(), entity("other", "unit", "Gretchin")]), query(
                "SELECT name FROM datasheets WHERE Id IN (:unit_id, :other_id) ORDER BY Id",
                {"unit_id": "000000001", "other_id": "000000004"}))
        self.assertEqual(result.rows, [("Warboss",), ("Gretchin",)], result)

    def test_runtime_authorizer_rejects_unsafe_access(self):
        unsafe = [
            ("DELETE FROM datasheets", {}),
            ("UPDATE datasheets SET name=:name", {"name": "changed"}),
            ("INSERT INTO datasheets (Id) VALUES (:id)", {"id": "new"}),
            ("DROP TABLE datasheets", {}),
            ("CREATE TABLE other (value TEXT)", {}),
            ("SELECT Id FROM datasheets; DELETE FROM datasheets", {}),
            ("SELECT Id FROM datasheets; SELECT Id FROM datasheets", {}),
            ("ATTACH DATABASE :path AS evil", {"path": ":memory:"}),
            ("DETACH DATABASE main", {}),
            ("PRAGMA user_version", {}),
            ("PRAGMA query_only = OFF", {}),
            ("SELECT load_extension(:path) FROM datasheets", {"path": "evil"}),
            ("SELECT randomblob(:size) FROM datasheets", {"size": 100}),
            ("SELECT name FROM sqlite_master", {}),
            ("SELECT COUNT(*) FROM sqlite_schema", {}),
            ("SELECT value FROM private_notes", {}),
            ("SELECT name FROM datasheets UNION SELECT value FROM private_notes", {}),
            ("WITH notes AS (SELECT value FROM private_notes) SELECT value FROM notes", {}),
            ("SELECT name FROM pragma_table_info(:table)", {"table": "datasheets"}),
            ("WITH RECURSIVE numbers(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM numbers WHERE n<3) "
             "SELECT n FROM numbers", {}),
        ]
        for sql, parameters in unsafe:
            with self.subTest(sql=sql), self.assertRaises(PolicyViolation):
                execute_bounded(self.path, sql, parameters)
        # Check the authorizer directly, independently of connection-level read-only mode.
        with closing(sqlite3.connect(self.path)) as connection:
            install_authorizer(connection)
            for sql in ("DELETE FROM datasheets", "PRAGMA user_version", "SELECT value FROM private_notes"):
                with self.subTest(sql=sql), self.assertRaises(sqlite3.DatabaseError):
                    connection.execute(sql).fetchall()
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM datasheets").fetchone(), (4,))
        with closing(open_readonly(self.path)) as connection:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("DELETE FROM datasheets")

    def test_sqlite_handles_ctes_columns_and_distinct_counts(self):
        sql = '''WITH infantry AS (
            SELECT DISTINCT DatasheetId FROM datasheets_keywords WHERE keyword=:keyword
        ) SELECT COUNT(DISTINCT d.Id) FROM datasheets d
          INNER JOIN infantry i ON i.DatasheetId=d.Id'''
        result = execute_bounded(self.path, sql, {"keyword": "INFANTRY"})
        self.assertEqual(result.rows, [(2,)])

    def test_numeric_guard_does_not_cast_symbolic_values_to_zero(self):
        expression = INTEGER_TEMPLATE.format(column="m.t")
        sql = f"SELECT {expression} AS value FROM datasheets_models m"
        result = execute_bounded(self.path, sql, {})
        self.assertEqual([row[0] for row in result.rows], [10, 9, None, None, -2, None])

    def test_generated_points_and_card_ability_rules(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.executescript('''
                CREATE TABLE datasheets_models_cost (DatasheetId TEXT, description TEXT, cost TEXT);
                INSERT INTO datasheets_models_cost VALUES
                    ('000000001', '1 model', '80'), ('000000001', '2 models', '160');
                CREATE TABLE abilities (Id TEXT, name TEXT, description TEXT);
                INSERT INTO abilities VALUES ('core', 'Catalog ability', 'Catalog rules');
                CREATE TABLE datasheets_abilities (DatasheetId TEXT, abilityId TEXT, name TEXT, description TEXT);
                INSERT INTO datasheets_abilities VALUES
                    ('000000001', 'core', '', ''),
                    ('000000001', 'core', 'Override', 'Card rules'),
                    ('000000001', '', 'Card only', 'Standalone rules');
            ''')
            connection.commit()
        result, _ = self.retrieve("Warboss points for 2 models", intent(entities=[entity()]), query(
            "SELECT description, cost FROM datasheets_models_cost "
            "WHERE DatasheetId=:unit_id AND description=:configuration",
            {"unit_id": "000000001", "configuration": "2 models"}))
        self.assertEqual(result.rows, [("2 models", "160")])
        result, _ = self.retrieve("Warboss ability names and text", intent(entities=[entity()]), query(
            "SELECT DISTINCT COALESCE(NULLIF(da.name, ''), a.name) AS ability_name, "
            "COALESCE(NULLIF(da.description, ''), a.description) AS ability_text "
            "FROM datasheets_abilities da LEFT JOIN abilities a ON a.Id=da.abilityId "
            "WHERE da.DatasheetId=:unit_id", {"unit_id": "000000001"}))
        self.assertEqual(set(result.rows), {
            ("Catalog ability", "Catalog rules"), ("Override", "Card rules"),
            ("Card only", "Standalone rules"),
        })

    def test_invalid_sql_and_parameter_boundaries(self):
        for sql in (None, "", "-- no query", "SELECT name FROM datasheets " + " " * (16 * 1024), "\ud800"):
            with self.subTest(sql_type=type(sql)), self.assertRaises(PolicyViolation):
                execute_bounded(self.path, sql, {})
        for parameters in (None, [], {"value": True}, {"value": float("nan")},
                           {"value": 2**63}, {"value": "x" * (16 * 1024 + 1)},
                           {"value": "\ud800"}, {"bad-key": "value"}):
            with self.subTest(parameters=parameters), self.assertRaises(PolicyViolation):
                execute_bounded(self.path, "SELECT :value FROM datasheets", parameters)

    def test_result_limits_and_execution_budgets(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.executemany("INSERT INTO datasheets (Id, name, factionId, role, legend, link) "
                                   "VALUES (?, ?, 'ORK', '', '', '')",
                                   [(f"extra-{number:03}", str(number)) for number in range(250)])
            connection.commit()
        for limit, count, truncated in [("", 200, True), (" LIMIT 3", 3, False)]:
            with self.subTest(limit=limit):
                result, _ = self.retrieve("List units", intent(), query(
                    "SELECT Id FROM datasheets ORDER BY Id DESC" + limit))
                self.assertEqual(result.status, "ok", result)
                self.assertEqual((len(result.rows), result.truncated), (count, truncated))
                self.assertEqual(result.rows[0], ("extra-249",))
        bounded = execute_bounded(self.path, "SELECT Id FROM datasheets", {},
                                  limits=ExecutionLimits(bytes=50))
        self.assertTrue(bounded.truncated)
        self.assertLessEqual(len(json.dumps(bounded.columns).encode()) +
                             len(json.dumps(bounded.rows).encode()), 50)
        count = execute_bounded(self.path, "SELECT COUNT(*) FROM datasheets", {})
        self.assertEqual(count.rows, [(254,)])
        with self.assertRaises(RetrievalFailure) as error:
            execute_bounded(self.path, "SELECT :value FROM datasheets", {"value": "x" * 2000},
                            limits=ExecutionLimits(bytes=64))
        self.assertEqual(error.exception.code, "result_budget")
        for limits in (ExecutionLimits(vm_steps=10), ExecutionLimits(seconds=0)):
            with self.subTest(limits=limits), self.assertRaises(RetrievalFailure) as error:
                execute_bounded(self.path, "SELECT COUNT(*) FROM datasheets a, datasheets b",
                                {}, limits=limits)
            self.assertEqual(error.exception.code, "query_budget")
        with closing(sqlite3.connect(self.path, timeout=0.01)) as connection:
            connection.execute("BEGIN EXCLUSIVE")
            connection.rollback()


if __name__ == "__main__":
    unittest.main()
