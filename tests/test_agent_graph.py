"""Offline evidence-loop checks using real LangGraph checkpoints and SQLite."""

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage

from agent import EvidenceContext, build_evidence_graph
from agent.graph import SQL_CONTEXT_MAX_BYTES, SQL_CONTEXT_MAX_ROWS
from retrieval.rulebook_retriever import RetrievedChunk, RetrievedRulebookContext, RulebookRetriever
from sql_retriever import RetrievalResult, retrieve_sql
from test_agent_nodes import planning_output
from test_rulebook_retriever import FakeBM25, FakeCollection, FakeEmbedding, FakeReranker, records
from test_sql_hybrid import client_for, entity, intent


REFINED = {
    "semantic_query": "How does the Feel No Pain ability prevent wounds?",
    "keyword_query": ["Feel No Pain", "wound", "damage"],
}


def rulebook_stub(*, empty=False):
    retriever = Mock()
    retriever.retrieve.side_effect = lambda query: RetrievedRulebookContext(
        query.normalised(), () if empty else (
            RetrievedChunk({"chunk_id": "r-001", "text": "Rule evidence", "pdf_page_numbers": [7]},
                           1, 2, None, 0.03, 0.9),
        ),
    )
    return retriever


class EvidenceGraphTests(unittest.TestCase):
    def setUp(self):
        self.graph = build_evidence_graph()
        self.config = {"configurable": {"thread_id": "session-1"}}
        self.sql = Mock(return_value=RetrievalResult(
            "ok", "fixed", ["id", "ability"], [("000000001", "Feel No Pain 5+")],
            sql="SELECT Id, ability FROM datasheets WHERE Id=:id", parameters={"id": "000000001"},
        ))
        self.rules = rulebook_stub()
        self.context = EvidenceContext(rulebook_retriever=self.rules, sql_retriever=self.sql)

    def invoke(self, *outputs, question="单位能力与相关规则？", run_id="run-1", **state):
        self.context.client = client_for(*outputs)
        return self.graph.invoke({"user_input": question, "run_id": run_id, **state},
                                 self.config, context=self.context)

    def test_routes_stream_actual_nodes_and_preserve_source_specific_inputs(self):
        for route, dependent in [("sql", False), ("rulebook", False), ("both", False), ("both", True)]:
            with self.subTest(route=route, dependent=dependent):
                self.setUp()
                output = planning_output(route, depends_on_sql=dependent)
                self.context.client = client_for(output, *([REFINED] if dependent else []))
                updates = list(self.graph.stream(
                    {"user_input": "  原始问题  ", "run_id": "run-1"}, self.config,
                    context=self.context, stream_mode="updates",
                ))
                names = [next(iter(update)) for update in updates]
                expected = ["prepare_turn", "plan"]
                if route in ("sql", "both"):
                    expected.append("retrieve_sql")
                    self.sql.assert_called_once_with(output["sql_question"])
                else:
                    self.sql.assert_not_called()
                if dependent:
                    expected.append("prepare_rulebook_query")
                    payload = json.loads(self.context.client.models.generate_content.call_args.kwargs["contents"])
                    self.assertEqual(payload["sql_evidence"]["rows"], [["000000001", "Feel No Pain 5+"]])
                    self.assertEqual(
                        self.context.client.models.generate_content.call_args.kwargs["config"]["temperature"],
                        0,
                    )
                if route in ("rulebook", "both"):
                    expected.append("retrieve_rulebook")
                    query = self.rules.retrieve.call_args.args[0]
                    self.assertEqual(query.original_query, "  原始问题  ")
                    self.assertEqual(query.semantic_query, REFINED["semantic_query"] if dependent else output["semantic_query"])
                else:
                    self.rules.retrieve.assert_not_called()
                self.assertEqual(names, expected + ["finish_evidence"])
                result = self.graph.get_state(self.config).values
                self.assertEqual(result["evidence_status"], "ok")
                self.assertIsNone(result["answer"])
                self.assertIsNone(result["outcome"])
                self.assertEqual(result["messages"], [])
                self.assertEqual(self.context.client.models.generate_content.call_count, 2 if dependent else 1)

    def test_real_sqlite_and_rulebook_pipeline_are_connected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.sqlite"
            with closing(sqlite3.connect(path)) as connection:
                connection.executescript('''
                    CREATE TABLE factions (Id TEXT, name TEXT);
                    CREATE TABLE source (Id TEXT, name TEXT);
                    CREATE TABLE datasheets (Id TEXT, name TEXT, factionId TEXT,
                        role TEXT, legend TEXT, link TEXT, sourceId TEXT DEFAULT '');
                    INSERT INTO factions VALUES ('ORK', 'Orks');
                    INSERT INTO datasheets (Id, name, factionId, role, legend, link)
                    VALUES ('000000001', 'Warboss', 'ORK', '', '', 'source');
                ''')
            self.context.sql_retriever = None
            self.context.rulebook_retriever = RulebookRetriever(
                collection=FakeCollection(), bm25=FakeBM25(),
                bm25_chunk_ids=["chunk-1", "chunk-2", "chunk-3"], chunk_records=records(),
                embedding_model=FakeEmbedding(), reranker=FakeReranker(),
            )
            with patch("agent.graph.retrieve_sql", wraps=partial(retrieve_sql, db_path=path)) as sql:
                result = self.invoke(planning_output("both", sql_question="Get Warboss metadata"),
                                     intent("fixed", "get_unit_all_metadata", {"d_id": "@unit"}, [entity()]))
            self.assertFalse(sql.call_args.kwargs["timing"])
            self.assertEqual(result["sql_result"].rows[0][:2], ("000000001", "Warboss"))
            self.assertEqual(result["sql_result"].parameters, {"d_id": "000000001"})
            self.assertEqual(result["sql_result"].strategy, "fixed")
            self.assertTrue(result["rulebook_result"].results)
            self.assertEqual(result["evidence_status"], "ok")

    def test_both_sql_failure_and_clarification_branch_behavior(self):
        for dependent in (False, True):
            for status in ("empty", "error", "unsupported", "needs_clarification"):
                with self.subTest(dependent=dependent, status=status):
                    self.setUp()
                    self.sql.return_value = RetrievalResult(status, message="Pick a unit.")
                    result = self.invoke(planning_output("both", depends_on_sql=dependent))
                    should_retrieve = not dependent and status != "needs_clarification"
                    self.assertEqual(self.rules.retrieve.call_count, int(should_retrieve))
                    self.assertEqual(result["evidence_status"], "partial" if should_retrieve else status)
                    self.context.client.models.generate_content.assert_called_once()
                    self.sql.assert_called_once()

    def test_pending_candidates_survive_checkpoint_and_reconstruct_whole_request(self):
        candidates = [{"id": "000000001", "name": "Boyz"}, {"id": "000000002", "name": "Boyz"}]
        whole_question = "Get Boyz abilities and explain the referenced Core Rules."
        self.sql.return_value = RetrievalResult("needs_clarification", candidates=candidates, message="Which Boyz?")
        first = self.invoke(planning_output("both", standalone_question=whole_question, depends_on_sql=True))
        self.assertEqual(first["pending_clarification"], {
            "question": whole_question, "message": "Which Boyz?", "candidates": candidates, "source_run_id": "run-1",
        })
        self.rules.retrieve.assert_not_called()
        self.sql.return_value = RetrievalResult("ok", columns=["ability"], rows=[("Feel No Pain 5+",)])
        question = "Get abilities for Boyz ID 000000002 and explain the referenced Core Rules."
        second = self.invoke(planning_output(
            "both", standalone_question=question, sql_question="Get abilities for Boyz ID 000000002", depends_on_sql=True,
        ), REFINED, question="第二个", run_id="run-2")
        planning_payload = json.loads(self.context.client.models.generate_content.call_args_list[0].kwargs["contents"])
        self.assertEqual(planning_payload["pending_clarification"], first["pending_clarification"])
        self.sql.assert_called_with("Get abilities for Boyz ID 000000002")
        self.assertEqual(second["turn_index"], 2)
        self.assertEqual(second["evidence_status"], "ok")
        self.assertIsNone(second["pending_clarification"])

    def test_empty_rulebook_and_unavailable_source_are_distinct(self):
        for route in ("rulebook", "both"):
            for failed in (False, True):
                with self.subTest(route=route, failed=failed):
                    self.setUp()
                    self.context.rulebook_retriever = rulebook_stub(empty=True)
                    if failed:
                        self.context.rulebook_retriever.retrieve.side_effect = FileNotFoundError("Missing index")
                    result = self.invoke(planning_output(route))
                    expected = "partial" if route == "both" else "error" if failed else "empty"
                    self.assertEqual(result["evidence_status"], expected)
                    self.assertEqual(bool(result["retrieval_errors"]), failed)
                    self.context.rulebook_retriever.retrieve.assert_called_once()

    def test_partial_rulebook_success_does_not_discard_unresolved_sql_candidates(self):
        pending = {
            "question": "Boyz stats and Advance rules", "message": "Which Boyz?",
            "candidates": [{"id": "000000002", "name": "Boyz"}], "source_run_id": "run-0",
        }
        self.sql.return_value = RetrievalResult("error", error_code="database_error")
        result = self.invoke(planning_output("both"), pending_clarification=pending)
        self.assertEqual(result["evidence_status"], "partial")
        self.assertEqual(result["pending_clarification"], pending)

    def test_checkpoint_roundtrip_restores_domain_types_and_isolates_sessions(self):
        self.sql.return_value.rows.append(("000000002", b"\x00\xff"))
        result = self.invoke(planning_output("both"), messages=[
            HumanMessage(content="Previous question"), AIMessage(content="Previous answer"),
        ])
        restored = self.graph.get_state(self.config).values
        self.assertEqual(restored, result)
        self.assertIsInstance(restored["sql_result"], RetrievalResult)
        self.assertIsInstance(restored["sql_result"].rows[0], tuple)
        self.assertIsInstance(restored["rulebook_result"].results, tuple)
        self.assertIsInstance(restored["rulebook_result"].results[0], RetrievedChunk)
        self.assertIsInstance(restored["route_plan"]["rulebook_query_set"].keyword_query, tuple)
        self.assertEqual(restored["sql_result"], self.sql.return_value)
        self.assertNotIn("client", restored)
        next_turn = self.invoke(planning_output("rulebook"), run_id="run-2")
        self.assertIsNone(next_turn["sql_result"])
        self.assertEqual(next_turn["turn_index"], 2)
        self.config = {"configurable": {"thread_id": "session-2"}}
        fresh = self.invoke(planning_output("direct"))
        self.assertEqual(fresh["turn_index"], 1)
        self.assertEqual(fresh["messages"], [])
        self.assertIsNone(fresh["sql_result"])

    def test_nonretrieval_routes_stop_at_evidence_without_answering(self):
        for route, status in [("direct", "skipped"), ("clarify", "needs_clarification"), ("unsupported", "unsupported")]:
            with self.subTest(route=route):
                result = self.invoke(planning_output(route))
                self.assertEqual(result["evidence_status"], status)
                self.assertIsNone(result["answer"])
                self.assertIsNone(result["outcome"])
                self.sql.assert_not_called()
                self.rules.retrieve.assert_not_called()

    def test_refinement_context_limits_do_not_mutate_sql_evidence(self):
        cases = [
            ([(str(number), "Feel No Pain") for number in range(60)], False, SQL_CONTEXT_MAX_ROWS),
            ([("000000001", "中" * 6000), ("000000002", "文" * 6000)], False, 1),
            ([("000000001", b"\x00\xff")], True, 1),
        ]
        for rows, source_truncated, expected_count in cases:
            with self.subTest(expected_count=expected_count):
                self.setUp()
                self.sql.return_value = RetrievalResult("ok", columns=["id", "ability"], rows=rows, truncated=source_truncated)
                result = self.invoke(planning_output("both", depends_on_sql=True), REFINED)
                payload = json.loads(self.context.client.models.generate_content.call_args.kwargs["contents"])["sql_evidence"]
                self.assertEqual(len(payload["rows"]), expected_count)
                self.assertLessEqual(len(json.dumps(payload, ensure_ascii=False).encode()), SQL_CONTEXT_MAX_BYTES)
                self.assertEqual(payload["result_truncated"], source_truncated)
                self.assertEqual(payload["context_truncated"], len(rows) > expected_count)
                self.assertEqual(result["sql_context_truncated"], len(rows) > expected_count)
                self.assertEqual(result["sql_result"].rows, rows)
                self.assertEqual(result["evidence_status"], "partial")

    def test_dependent_route_with_no_usable_sql_rows_does_not_guess(self):
        for rows, truncated in [([], True), ([("x" * SQL_CONTEXT_MAX_BYTES, "ability")], False)]:
            with self.subTest(truncated=truncated):
                self.setUp()
                self.sql.return_value = RetrievalResult("ok", rows=rows, truncated=truncated)
                result = self.invoke(planning_output("both", depends_on_sql=True))
                self.rules.retrieve.assert_not_called()
                self.context.client.models.generate_content.assert_called_once()
                self.assertEqual(result["evidence_status"], "partial")
                self.assertIsNone(result["rulebook_result"])

    def test_invalid_refinement_keeps_sql_checkpoint_and_does_not_retry(self):
        for response in [{"semantic_query": "", "keyword_query": ["bad"]}, RuntimeError("Provider unavailable")]:
            with self.subTest(response=response):
                self.setUp()
                self.context.client = client_for(planning_output("both", depends_on_sql=True), response if isinstance(response, dict) else {})
                if isinstance(response, Exception):
                    self.context.client.models.generate_content.side_effect = [
                        SimpleNamespace(text=json.dumps(planning_output("both", depends_on_sql=True))), response,
                    ]
                with self.assertRaises((ValueError, RuntimeError)):
                    self.graph.invoke({"user_input": "Question", "run_id": "run-1"}, self.config, context=self.context)
                self.rules.retrieve.assert_not_called()
                self.assertEqual(self.context.client.models.generate_content.call_count, 2)
                restored = self.graph.get_state(self.config).values
                self.assertEqual(restored["sql_result"], self.sql.return_value)
                self.assertIsNone(restored["evidence_status"])
                self.assertIsNone(restored["outcome"])

    def test_default_rulebook_is_lazy_and_reused_and_old_errors_are_reset(self):
        self.context.rulebook_retriever = None
        with patch("agent.graph.RulebookRetriever.from_paths", return_value=self.rules) as load:
            self.invoke(planning_output("sql"))
            load.assert_not_called()
            self.rules.retrieve.side_effect = OSError("Index unavailable")
            self.invoke(planning_output("rulebook"), run_id="run-2")
            self.rules.retrieve.side_effect = rulebook_stub().retrieve.side_effect
            result = self.invoke(planning_output("rulebook"), run_id="run-3")
            load.assert_called_once()
        self.assertEqual(result["retrieval_errors"], {})
        self.assertEqual(result["evidence_status"], "ok")


if __name__ == "__main__":
    unittest.main()
