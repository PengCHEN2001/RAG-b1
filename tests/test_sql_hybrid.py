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
from sql_retriever.result import PolicyViolation, RetrievalFailure
from sql_retriever.safety import (
    ExecutedQuery, ExecutionLimits, execute_bounded, install_authorizer, validate_sql,
)
from sql_retriever.schema import inspect_schema, integer_expression, open_readonly


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
                CREATE TABLE datasheets (Id TEXT, name TEXT, factionId TEXT,
                    role TEXT, legend TEXT, link TEXT);
                CREATE TABLE datasheets_keywords (DatasheetId TEXT, keyword TEXT);
                CREATE TABLE datasheets_models (DatasheetId TEXT, name TEXT, t TEXT);
                CREATE TABLE private_notes (value TEXT);
                INSERT INTO factions VALUES ('ORK', 'Orks'), ('TYR', 'Tyranids');
                INSERT INTO datasheets VALUES
                    ('000000001', 'Warboss', 'ORK', '', '', ''),
                    ('000000002', 'Boyz', 'ORK', '', '', ''),
                    ('000000003', 'Boyz', 'TYR', '', '', ''),
                    ('000000004', 'Gretchin', 'ORK', '', '', '');
                INSERT INTO datasheets_keywords VALUES
                    ('000000001', 'INFANTRY'), ('000000001', 'INFANTRY'),
                    ('000000002', 'INFANTRY');
                INSERT INTO datasheets_models VALUES
                    ('000000001', 'Warboss', '10'), ('000000002', 'Boyz', '9'),
                    ('000000003', 'Dice', 'D6'), ('000000004', 'Empty', ''),
                    ('000000001', 'Signed', '-2'), ('000000002', 'Save', '2+');
            ''')
            connection.commit()
        self.schema = inspect_schema(self.path)

    def retrieve(self, question, *responses):
        client = client_for(*responses)
        return retrieve_sql(question, client=client, db_path=self.path), client

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
        for name, count in [("Boyz", 2), ("Warbos", 0)]:
            with self.subTest(name=name):
                result, client = self.retrieve(f"Get {name} metadata", intent(
                    "fixed", "get_unit_all_metadata", {"d_id": "@unit"}, [entity(value=name)]))
                self.assertEqual(result.status, "needs_clarification")
                self.assertEqual(len(result.candidates), count)
                self.assertEqual(client.models.generate_content.call_count, 1)

    def test_generated_route_executes_grounded_numeric_query(self):
        sql = f'''SELECT d.Id, d.name FROM datasheets d
            INNER JOIN datasheets_models m ON m.DatasheetId=d.Id
            WHERE d.factionId=:faction_id AND {integer_expression("m.t")}>=:minimum
            AND EXISTS (SELECT k.DatasheetId FROM datasheets_keywords k
                        WHERE k.DatasheetId=d.Id AND k.keyword=:keyword)'''
        result, client = self.retrieve("Orks INFANTRY with toughness at least 10", intent(
            entities=[entity("faction", "faction", "Orks")]),
            query(sql, {"faction_id": "ORK", "minimum": 10, "keyword": "INFANTRY"}))
        self.assertEqual((result.status, result.strategy), ("ok", "generated"), result)
        self.assertEqual(result.rows, [("000000001", "Warboss")])
        self.assertEqual(client.models.generate_content.call_count, 2)

    def test_compilation_repair_is_limited_to_one_attempt(self):
        invalid = query("SELECT imaginary FROM datasheets")
        for correction, expected in [(query("SELECT name FROM datasheets"), "ok"), (invalid, "error")]:
            with self.subTest(expected=expected):
                result, client = self.retrieve("List unit names", intent(), invalid, correction)
                self.assertEqual(result.status, expected)
                self.assertEqual(client.models.generate_content.call_count, 3)
                if expected == "error":
                    self.assertEqual(result.error_code, "sql_compilation")

    def test_empty_policy_and_budget_failures_do_not_retry(self):
        cases = [
            (query("SELECT Id FROM datasheets WHERE name=:name", {"name": "Nobody"}), "empty", None),
            (query("DELETE FROM datasheets"), "error", "sql_policy"),
        ]
        for response, status, code in cases:
            with self.subTest(status=status):
                result, client = self.retrieve("Find units", intent(), response)
                self.assertEqual((result.status, result.error_code), (status, code))
                self.assertEqual(client.models.generate_content.call_count, 2)
        client = client_for(intent(), query("SELECT Id FROM datasheets"))
        with patch("sql_retriever.hybrid.execute_bounded", side_effect=[
                ExecutedQuery([], [], False), RetrievalFailure("query_budget", "Budget exceeded.")]):
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
        result = retrieve_sql("Get units", client=client, db_path=self.path)
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
            validate_sql("SELECT Id FROM datasheets WHERE Id=:id", {}, self.schema)
        result, _ = self.retrieve("Warboss model stats", intent(entities=[entity()]), query(
            "SELECT name FROM datasheets_models WHERE DatasheetId=:unit_id", {"unit_id": "000000002"}))
        self.assertEqual(result.error_code, "sql_policy")

    def test_sql_validation_and_runtime_authorizer_reject_unsafe_access(self):
        unsafe = [
            ("DELETE FROM datasheets", {}),
            ("SELECT Id FROM datasheets; DELETE FROM datasheets", {}),
            ("ATTACH DATABASE :path AS evil", {"path": ":memory:"}),
            ("SELECT load_extension(:path) FROM datasheets", {"path": "evil"}),
            ("SELECT name FROM sqlite_master", {}),
            ("SELECT value FROM private_notes", {}),
            ("SELECT d.Id FROM datasheets d CROSS JOIN factions f", {}),
            ("SELECT CAST(t AS INTEGER) FROM datasheets_models", {}),
        ]
        for sql, parameters in unsafe:
            with self.subTest(sql=sql), self.assertRaises(PolicyViolation):
                validate_sql(sql, parameters, self.schema)
        # Runtime protection must work even if AST validation is accidentally bypassed.
        with closing(open_readonly(self.path)) as connection:
            install_authorizer(connection, self.schema)
            for sql in ("DELETE FROM datasheets", "PRAGMA user_version", "SELECT value FROM private_notes"):
                with self.subTest(sql=sql), self.assertRaises(sqlite3.DatabaseError):
                    connection.execute(sql).fetchall()
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM datasheets").fetchone(), (4,))

    def test_numeric_guard_does_not_cast_symbolic_values_to_zero(self):
        expression = integer_expression("m.t")
        sql = validate_sql(f"SELECT {expression} AS value FROM datasheets_models m", {}, self.schema)
        result = execute_bounded(self.path, sql, {}, self.schema)
        self.assertEqual([row[0] for row in result.rows], [10, 9, None, None, -2, None])
        with self.assertRaises(PolicyViolation):
            validate_sql("SELECT t FROM datasheets_models WHERE t>:minimum", {"minimum": 9}, self.schema)

    def test_result_limits_and_execution_budgets(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.executemany("INSERT INTO datasheets VALUES (?, ?, 'ORK', '', '', '')",
                                   [(f"extra-{number:03}", str(number)) for number in range(250)])
            connection.commit()
        for limit, count, truncated in [("", 200, True), (" LIMIT 3", 3, False)]:
            with self.subTest(limit=limit):
                result, _ = self.retrieve("List units", intent(), query(
                    "SELECT Id FROM datasheets ORDER BY Id DESC" + limit))
                self.assertEqual(result.status, "ok", result)
                self.assertEqual((len(result.rows), result.truncated), (count, truncated))
                self.assertEqual(result.rows[0], ("extra-249",))
        bounded = execute_bounded(self.path, "SELECT Id FROM datasheets", {}, self.schema,
                                  limits=ExecutionLimits(bytes=25))
        self.assertTrue(bounded.truncated)
        for limits in (ExecutionLimits(vm_steps=10), ExecutionLimits(seconds=0)):
            with self.subTest(limits=limits), self.assertRaises(RetrievalFailure) as error:
                execute_bounded(self.path, "SELECT COUNT(*) FROM datasheets a, datasheets b",
                                {}, self.schema, limits=limits)
            self.assertEqual(error.exception.code, "query_budget")
        with closing(sqlite3.connect(self.path, timeout=0.01)) as connection:
            connection.execute("BEGIN EXCLUSIVE")
            connection.rollback()


if __name__ == "__main__":
    unittest.main()
