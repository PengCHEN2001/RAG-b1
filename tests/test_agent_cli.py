"""Offline checks for the real graph runner, CLI and persistent graph traces."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from agent import EvidenceContext, build_agent_graph
from agent.runtime import _TraceFile, read_trace, run_turn, source_details, trace_value
from sql_retriever import RetrievalResult
from test_agent_graph import REFINED, rulebook_stub
from test_agent_nodes import planning_output
from test_agent_response import response_output
from test_sql_hybrid import client_for
from ui.cli import main


class AgentCLITests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "traces"
        self.graph = build_agent_graph()
        self.sql = Mock(return_value=RetrievalResult(
            "ok", "fixed", ["unit_name", "ability"], [("Warboss", "Feel No Pain")],
            sql="SELECT name, ability FROM units WHERE id=:id", parameters={"id": "000000001"},
        ))
        self.rules = rulebook_stub()
        self.context = EvidenceContext(rulebook_retriever=self.rules, sql_retriever=self.sql)

    def invoke(self, *outputs, session="session-1", question="Warboss 的能力？", **options):
        self.context.client = client_for(*outputs)
        return run_turn(self.graph, self.context, session_id=session, question=question,
                        trace_dir=self.root, **options)

    def cli(self, argv, *, inputs=()):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr), patch("builtins.input", side_effect=inputs):
            code = main(argv, graph=self.graph, context=self.context)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_trace_contains_actual_node_inputs_results_timings_and_source_provenance(self):
        received = []
        result = self.invoke(planning_output("both", depends_on_sql=True), REFINED,
                             response_output("能力与规则。[S1] [R1]"), question="  原始问题  ", on_event=received.append)
        trace = read_trace(result["trace_path"])
        self.assertEqual(trace["events"], received)
        self.assertEqual(trace["outcome"], "completed")
        self.assertTrue(result["trace_complete"])
        self.assertEqual([event["seq"] for event in received], list(range(1, len(received) + 1)))
        self.assertEqual(sum(event["event"] == "run.finished" for event in received), 1)
        self.assertEqual(received[0]["payload"]["user_input"], "  原始问题  ")
        self.assertEqual({event["run_id"] for event in received}, {result["run_id"]})
        self.assertEqual({event["session_id"] for event in received}, {"session-1"})
        starts = [event for event in received if event["event"] == "step.started"]
        self.assertEqual([event["node"] for event in starts], [
            "prepare_turn", "plan", "retrieve_sql", "prepare_rulebook_query", "retrieve_rulebook",
            "finish_evidence", "respond", "finalize",
        ])
        self.assertTrue(all(event["parent_span_id"] == result["run_id"] for event in starts))
        self.assertTrue(all(value >= 0 for value in result["step_durations_ms"].values()))
        self.assertGreaterEqual(result["duration_ms"], 0)
        planned = next(event for event in received if event["event"] == "route.selected")
        self.assertTrue(planned["payload"]["route_plan"]["depends_on_sql"])
        self.assertTrue(any(event["event"] == "query.refined" for event in received))
        terminal = received[-1]["payload"]
        self.assertEqual(terminal["state"]["sql_result"]["parameters"], {"id": "000000001"})
        self.assertEqual(terminal["state"]["rulebook_result"]["results"][0]["chunk"]["pdf_page_numbers"], [7])
        self.assertIsNone(terminal["usage"]["total_token_count"])
        self.assertFalse(terminal["usage"]["complete"])
        self.assertEqual(terminal["llm_calls"], 3)
        metadata = json.loads((result["trace_path"].parent / "session.json").read_text())
        self.assertEqual(metadata["trace_scope"], "graph")
        self.assertIsNone(metadata["retriever_internals"])
        self.assertEqual(len(metadata["uv_lock_sha256"]), 64)

    def test_followups_use_same_memory_but_distinct_retained_run_files(self):
        first = self.invoke(planning_output(), response_output())
        second = self.invoke(planning_output(), response_output(), question="它呢？")
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(second["state"]["turn_index"], 2)
        records = read_trace(second["trace_path"])["events"]
        self.assertEqual(len(records[0]["payload"]["previous_messages"]), 2)
        plan_start = next(event for event in records if event["event"] == "step.started" and event["node"] == "plan")
        self.assertEqual(len(plan_start["payload"]["input"]["messages"]), 2)
        self.assertEqual(len(list(first["trace_path"].parent.glob("*.jsonl"))), 2)

    def test_graph_failure_records_node_stack_once_and_preserves_old_history(self):
        first = self.invoke(planning_output(), response_output())
        result = self.invoke(planning_output(), response_output("Unknown reference. [S99]"))
        self.assertEqual(result["outcome"], "error")
        self.assertEqual(result["failed_node"], "respond")
        self.assertEqual(result["state"]["messages"], first["state"]["messages"])
        self.assertIsNone(result["state"]["answer"])
        events = read_trace(result["trace_path"])["events"]
        self.assertEqual(sum(event["event"] == "run.finished" for event in events), 1)
        self.assertEqual([event["node"] for event in events if event["event"] == "step.failed"], ["respond"])
        self.assertEqual(events[-1]["payload"]["error"]["type"], "ValueError")
        self.assertTrue(events[-1]["payload"]["error"]["stack"])
        self.assertNotIn("source", events[-1]["payload"]["error"]["stack"][0])

    def test_cancelled_turn_finishes_once_and_does_not_continue_retrieval(self):
        self.context.client = client_for({})
        self.context.client.models.generate_content.side_effect = KeyboardInterrupt()
        result = run_turn(self.graph, self.context, session_id="cancel", question="Question", trace_dir=self.root)
        self.assertEqual(result["outcome"], "cancelled")
        self.assertIsNone(result["state"]["answer"])
        self.assertEqual(result["state"]["messages"], [])
        self.sql.assert_not_called()
        events = read_trace(result["trace_path"])["events"]
        self.assertEqual(sum(event["event"] == "run.finished" for event in events), 1)
        self.assertEqual(events[-1]["payload"]["outcome"], "cancelled")

    def test_provider_errors_and_configured_keys_are_not_written_or_sent_to_viewers(self):
        self.context.client = client_for({})
        self.context.client.api_key = "injected-private-key"
        self.context.client.models.generate_content.side_effect = RuntimeError("Authorization: Bearer foreign-secret")
        received = []
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "configured-private-key"}):
            result = run_turn(self.graph, self.context, session_id="private", question="configured-private-key injected-private-key",
                              trace_dir=self.root, on_event=received.append)
        content = result["trace_path"].read_text()
        self.assertNotIn("configured-private-key", content)
        self.assertNotIn("injected-private-key", content)
        self.assertNotIn("foreign-secret", content)
        self.assertNotIn("foreign-secret", json.dumps(received))
        self.assertIn("[REDACTED]", content)
        self.assertEqual(trace_value({"Authorization": "Bearer secret"}), {"Authorization": "[REDACTED]"})

    def test_blobs_are_labeled_without_losing_sql_values(self):
        self.sql.return_value.rows = [("000000001", b"\x00\xff")]
        result = self.invoke(planning_output(), response_output())
        state = read_trace(result["trace_path"])["events"][-1]["payload"]["state"]
        self.assertEqual(state["sql_result"]["rows"][0][1], {"$bytes_hex": "00ff"})
        self.assertEqual(result["state"]["sql_result"].rows[0][1], b"\x00\xff")

    def test_trace_open_and_write_failures_warn_but_do_not_lose_valid_answers(self):
        for opening in (True, False):
            with self.subTest(opening=opening):
                self.context.client = client_for(planning_output(), response_output())
                received = []
                if opening:
                    patcher = patch("pathlib.Path.mkdir", side_effect=PermissionError("private path"))
                else:
                    def broken_write(writer, event):
                        writer._fail(OSError("Disk full"))
                    patcher = patch.object(_TraceFile, "write", broken_write)
                with patcher:
                    result = run_turn(self.graph, self.context, session_id=f"io-{opening}", question="Warboss?",
                                      trace_dir=self.root, on_event=received.append)
                self.assertEqual(result["outcome"], "completed")
                self.assertTrue(result["state"]["answer"])
                self.assertFalse(result["trace_complete"])
                self.assertEqual(sum(event["event"] == "trace.failed" for event in received), 1)
                self.assertEqual(sum(event["event"] == "run.finished" for event in received), 1)

    def test_incomplete_tail_and_missing_terminal_event_are_reported_without_execution(self):
        result = self.invoke(planning_output(), response_output())
        with result["trace_path"].open("a") as handle:
            handle.write('{"incomplete":')
        trace = read_trace(result["trace_path"])
        self.assertEqual(trace["outcome"], "completed")
        self.assertTrue(trace["warnings"])
        unfinished = result["trace_path"].parent / "interrupted.jsonl"
        unfinished.write_text(json.dumps(trace["events"][0]) + '\n{"unfinished":')
        self.assertEqual(read_trace(unfinished)["outcome"], "interrupted")
        bad = result["trace_path"].parent / "bad.jsonl"
        bad.write_text("not JSON\n")
        with self.assertRaises(ValueError):
            read_trace(bad)

    def test_single_question_cli_prints_answer_sources_steps_and_persists_by_default(self):
        self.context.client = client_for(planning_output("both"), response_output("答案。[S1] [R1]"))
        code, stdout, stderr = self.cli(["原始问题", "--trace", "--sources", "--trace-dir", str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn("答案。[S1] [R1]", stdout)
        self.assertIn('"000000001"', stdout)
        self.assertIn('"pdf_page_numbers"', stdout)
        self.assertIn("… plan", stderr)
        self.assertIn("route=both", stderr)
        self.assertIn("LLM calls=2", stderr)
        self.assertIn("total=null", stderr)
        self.assertIn("outcome=completed", stderr)
        self.assertEqual(len(list(self.root.glob("*/*.jsonl"))), 1)

    def test_no_trace_disables_disk_but_not_step_display(self):
        self.context.client = client_for(planning_output(), response_output())
        code, stdout, stderr = self.cli(["Warboss?", "--no-trace", "--trace", "--trace-dir", str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn("[S1]", stdout)
        self.assertIn("… plan", stderr)
        self.assertFalse(self.root.exists())

    def test_interactive_commands_keep_history_new_clears_memory_and_retains_traces(self):
        self.context.client = client_for(
            planning_output(), response_output(), planning_output(), response_output(),
            planning_output("clarify"), response_output("请说明单位。", insufficient=True),
        )
        code, stdout, stderr = self.cli(["--trace-dir", str(self.root)], inputs=[
            "/trace", "Warboss?", "/sources S1", "Its abilities?", "/new", "/sources", "它呢？", "/quit",
        ])
        self.assertEqual(code, 0)
        self.assertIn('"query_id"', stdout)
        self.assertIn("{}", stdout)
        self.assertIn("New session:", stderr)
        self.assertIn("Session LLM calls=4", stderr)
        self.assertEqual(stderr.count("Session LLM calls=2"), 2)
        calls = self.context.client.models.generate_content.call_args_list
        self.assertEqual(len(json.loads(calls[2].kwargs["contents"])["history"]), 2)
        self.assertEqual(json.loads(calls[4].kwargs["contents"])["history"], [])
        paths = list(self.root.glob("*/*.jsonl"))
        self.assertEqual(len(paths), 3)
        self.assertEqual(len({path.parent for path in paths}), 2)
        old_session = next(path.parent.name for path in paths if read_trace(path)["events"][0]["payload"]["user_input"] == "Warboss?")
        self.assertEqual(self.graph.get_state({"configurable": {"thread_id": old_session}}).values, {})

    def test_offline_inspect_prints_full_records_or_sources_without_building_a_graph(self):
        result = self.invoke(planning_output(), response_output())
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("ui.cli.build_agent_graph") as build, redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["--inspect", str(result["trace_path"]), "--sources"])
            build.assert_not_called()
        self.assertEqual(code, 0)
        self.assertIn("outcome=completed", stdout.getvalue())
        self.assertIn('"query_id"', stdout.getvalue())
        with redirect_stdout(io.StringIO()) as full:
            self.assertEqual(main(["--inspect", str(result["trace_path"]), "--json"]), 0)
        self.assertEqual(json.loads(full.getvalue())["outcome"], "completed")
        self.assertEqual(source_details(result["state"])["S1"]["parameters"], {"id": "000000001"})

    def test_interactive_error_can_be_followed_by_a_fresh_successful_turn(self):
        self.context.client = client_for(planning_output(), response_output())
        responses = list(self.context.client.models.generate_content.side_effect)
        self.context.client.models.generate_content.side_effect = [RuntimeError("private"), *responses]
        code, stdout, stderr = self.cli(["--trace-dir", str(self.root)], inputs=[
            "Warboss abilities?", "Warboss faction?", "/quit",
        ])
        self.assertEqual(code, 0)
        self.assertIn("[S1]", stdout)
        self.assertIn("outcome=error", stderr)
        self.assertIn("outcome=completed", stderr)
        self.assertEqual(self.context.client.models.generate_content.call_count, 3)
        outcomes = {read_trace(path)["outcome"] for path in self.root.glob("*/*.jsonl")}
        self.assertEqual(outcomes, {"error", "completed"})

    def test_cli_failure_exit_codes_still_retain_a_terminal_trace(self):
        for failure, expected_code, outcome in [(RuntimeError("private diagnostic"), 1, "error"),
                                                 (KeyboardInterrupt(), 130, "cancelled")]:
            with self.subTest(outcome=outcome):
                self.context.client = client_for({})
                self.context.client.models.generate_content.side_effect = failure
                code, stdout, stderr = self.cli(["Warboss?", "--trace-dir", str(self.root / outcome)])
                self.assertEqual(code, expected_code)
                self.assertEqual(stdout, "")
                self.assertIn(f"{outcome} at plan: {type(failure).__name__}", stderr)
                self.assertNotIn("private diagnostic", stderr)
                path = next((self.root / outcome).glob("*/*.jsonl"))
                self.assertEqual(read_trace(path)["outcome"], outcome)


if __name__ == "__main__":
    unittest.main()
