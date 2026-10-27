"""Provider usage, SQL-internal calls and failed/cancelled turns without network."""

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from google.genai import types

from agent import EvidenceContext, build_agent_graph
from agent.runtime import read_trace, run_turn
from evaluation.adapters.trace_adapter import load_evaluation_trace
from llm.instrumentation import TOKEN_FIELDS, session_stats, summarize_usage
from sql_retriever import retrieve_sql
from sql_retriever.fixed_func import sql_retrieve_fixed_funcs
from test_agent_graph import REFINED, rulebook_stub
from test_agent_nodes import planning_output
from test_agent_response import response_output
from test_sql_hybrid import entity, intent, query


def usage_response(output, **changes):
    counts = dict(zip(TOKEN_FIELDS, (100, 10, 5, 20, 2, 117)))
    counts.update(changes)
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[
            types.Part(text=json.dumps(output)),
        ]))], usage_metadata=types.GenerateContentResponseUsageMetadata(**counts),
        model_version="test-model-version", response_id="test-response",
    )


class LLMUsageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.db = self.root / "rules.sqlite"
        with closing(sqlite3.connect(self.db)) as connection:
            connection.executescript("""
                CREATE TABLE factions (Id TEXT, name TEXT);
                CREATE TABLE datasheets (Id TEXT, name TEXT, factionId TEXT, role TEXT, legend TEXT, link TEXT);
                INSERT INTO factions VALUES ('ORK', 'Orks');
                INSERT INTO datasheets VALUES ('000000001', 'Warboss', 'ORK', '', '', '');
            """)
            connection.commit()
        self.graph = build_agent_graph()
        self.client = Mock()
        self.context = EvidenceContext(client=self.client, rulebook_retriever=rulebook_stub())

    def run_responses(self, *responses, session="usage", **options):
        self.client.models.generate_content.side_effect = list(responses)
        with patch("agent.graph.retrieve_sql", partial(retrieve_sql, db_path=self.db)):
            return run_turn(self.graph, self.context, session_id=session, question="Warboss",
                            trace_dir=self.root / "traces", **options)

    def test_provider_fields_parent_spans_and_eval_export_preserve_actual_totals(self):
        events = []
        first = self.run_responses(usage_response(planning_output("rulebook")),
                                   usage_response(response_output("规则。[R1]")), on_event=events.append)
        self.assertEqual(first["llm_calls"], 2)
        self.assertTrue(first["usage"]["complete"])
        for field, expected in zip(TOKEN_FIELDS, (200, 20, 10, 40, 4, 234)):
            self.assertEqual(first["usage"][field], expected)
        self.assertEqual(read_trace(first["trace_path"])["events"], events)
        node_ids = {e["node"]: e["span_id"] for e in events if e["event"] == "step.started"}
        llm_starts = [e for e in events if e["event"] == "llm.started"]
        self.assertEqual(len({e["span_id"] for e in llm_starts}), 2)
        for event in llm_starts:
            self.assertEqual(event["parent_span_id"], node_ids[event["node"]])
        self.assertNotIn("usage", first["state"])
        exported = load_evaluation_trace(first["trace_path"])
        self.assertEqual(exported["llm_calls"], 2)
        self.assertEqual(exported["usage"], first["usage"])
        second = self.run_responses(usage_response(planning_output("rulebook")),
                                    usage_response(response_output("规则。[R1]")))
        self.assertEqual(second["llm_calls"], 2)
        totals = session_stats([first, second])
        self.assertEqual(totals["llm_calls"], 4)
        self.assertEqual(totals["usage"]["total_token_count"], 468)

    def test_fixed_generated_repair_and_dependent_both_count_each_actual_sdk_call(self):
        fixed = intent("fixed", "get_unit_all_metadata", {"d_id": "@unit"}, [entity()])
        cases = [
            ([planning_output(), fixed, response_output()], 3, ["plan", "sql.route", "respond"]),
            ([planning_output(), intent(), query("SELECT name FROM datasheets"), response_output()],
             4, ["plan", "sql.route", "sql.generate", "respond"]),
            ([planning_output("both", depends_on_sql=True), intent(), query("SELECT imaginary FROM datasheets"),
              query("SELECT name FROM datasheets"), REFINED, response_output("说明。[S1] [R1]")],
             6, ["plan", "sql.route", "sql.generate", "sql.repair", "prepare_rulebook_query", "respond"]),
        ]
        for outputs, expected, purposes in cases:
            with self.subTest(purposes=purposes):
                result = self.run_responses(*(usage_response(output) for output in outputs))
                self.assertEqual(result["outcome"], "completed", result["error"])
                self.assertEqual(result["llm_calls"], expected)
                self.assertEqual(result["usage"]["total_token_count"], expected * 117)
                events = read_trace(result["trace_path"])["events"]
                self.assertEqual([e["payload"]["purpose"] for e in events if e["event"] == "llm.started"], purposes)

    def test_api_errors_record_null_and_partial_totals_do_not_masquerade_as_complete(self):
        failed = self.run_responses(RuntimeError("secret provider diagnostic"))
        self.assertEqual(failed["outcome"], "error")
        self.assertEqual(failed["llm_calls"], 1)
        self.assertTrue(all(failed["usage"][field] is None for field in TOKEN_FIELDS))
        records = read_trace(failed["trace_path"])["events"]
        event = next(e for e in records if e["event"] == "llm.failed")
        self.assertIsNone(event["payload"]["usage"])
        self.assertNotIn("secret provider diagnostic", failed["trace_path"].read_text())
        partial_result = self.run_responses(
            usage_response(planning_output()), RuntimeError("SQL provider failed"),
            usage_response(response_output("无法读取数据。", insufficient=True)),
        )
        self.assertEqual(partial_result["llm_calls"], 3)
        usage = partial_result["usage"]
        self.assertIsNone(usage["total_token_count"])
        self.assertEqual(usage["known_totals"]["total_token_count"], 234)
        self.assertEqual(usage["reported_calls"]["total_token_count"], 2)
        self.assertFalse(usage["complete"])
        self.assertIsNone(session_stats([failed, partial_result])["usage"]["total_token_count"])

    def test_invalid_model_output_still_records_received_usage_and_missing_fields_stay_null(self):
        result = self.run_responses(usage_response({"invalid": True}))
        self.assertEqual(result["outcome"], "error")
        self.assertEqual(result["llm_calls"], 1)
        self.assertEqual(result["usage"]["total_token_count"], 117)
        events = read_trace(result["trace_path"])["events"]
        self.assertLess(next(e["seq"] for e in events if e["event"] == "llm.finished"),
                        next(e["seq"] for e in events if e["event"] == "step.failed"))
        result = self.run_responses(
            usage_response(planning_output("rulebook"), prompt_token_count=None),
            usage_response(response_output("规则。[R1]")),
        )
        self.assertIsNone(result["usage"]["prompt_token_count"])
        self.assertEqual(result["usage"]["known_totals"]["prompt_token_count"], 100)
        self.assertEqual(result["usage"]["total_token_count"], 234)
        self.assertFalse(result["usage"]["complete"])
        missing = self.run_responses(
            usage_response(planning_output("rulebook")),
            SimpleNamespace(text=json.dumps(response_output("规则。[R1]"))),
        )
        self.assertIsNone(missing["usage"]["total_token_count"])
        self.assertEqual(missing["usage"]["known_totals"]["total_token_count"], 117)

    def test_cancelled_inflight_response_keeps_usage_and_warm_retriever_is_reused(self):
        from threading import Event

        cancel = Event()

        def finish_call(**kwargs):
            cancel.set()
            return usage_response(planning_output("rulebook"))

        self.client.models.generate_content.side_effect = finish_call
        events = []
        result = run_turn(self.graph, self.context, session_id="cancel", question="Warboss",
                          trace_dir=None, on_event=events.append, cancel_event=cancel)
        self.assertEqual(result["outcome"], "cancelled")
        self.assertEqual(result["llm_calls"], 1)
        self.assertEqual(result["usage"]["total_token_count"], 117)
        self.assertEqual(result["state"]["messages"], [])
        self.assertEqual(sum(e["event"] == "llm.finished" for e in events), 1)
        self.assertEqual(sum(e["event"] == "run.finished" for e in events), 1)
        self.context.rulebook_retriever = None
        rules = rulebook_stub()
        with patch("agent.graph.RulebookRetriever.from_paths", return_value=rules) as load:
            for _ in range(2):
                self.run_responses(usage_response(planning_output("rulebook")),
                                   usage_response(response_output("规则。[R1]")))
            load.assert_called_once()
        self.assertIs(self.context.rulebook_retriever, rules)

    def test_legacy_fixed_function_uses_same_wrapper_and_explicit_zero_is_preserved(self):
        events = []
        self.client.models.generate_content.return_value = usage_response(
            {"name": None, "arguments": {}}, **dict.fromkeys(TOKEN_FIELDS, 0))
        self.assertIsNone(sql_retrieve_fixed_funcs("Question", client=self.client, on_llm_event=events.append))
        usage = events[-1]["payload"]["usage"]
        self.assertTrue(all(value == 0 for value in usage.values()))
        self.assertEqual(summarize_usage([usage])["total_token_count"], 0)
        self.assertIsNone(summarize_usage([None])["total_token_count"])
        self.assertEqual(events[0]["payload"]["purpose"], "sql.fixed_function")


if __name__ == "__main__":
    unittest.main()
