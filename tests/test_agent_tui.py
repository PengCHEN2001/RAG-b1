"""Headless UI checks against the real graph with offline model/retriever stubs."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

from textual.widgets import Button, DataTable, Input, Select, Static, TabbedContent, TextArea

from agent import EvidenceContext, build_agent_graph
from agent.runtime import read_trace, run_turn
from sql_retriever import RetrievalResult
from test_agent_graph import rulebook_stub
from test_agent_nodes import planning_output
from test_agent_response import response_output
from test_sql_hybrid import client_for
from ui.tui import RAGApp, main


class AgentTUITests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "traces"
        self.graph = build_agent_graph()
        self.sql = Mock(return_value=RetrievalResult(
            "ok", "fixed", ["unit_name"], [("Warboss",)], sql="SELECT name FROM units",
        ))
        self.context = EvidenceContext(rulebook_retriever=rulebook_stub(), sql_retriever=self.sql)

    async def ask(self, app, pilot, question):
        app.query_one("#question", Input).value = question
        app.query_one("#question", Input).focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    async def test_chat_sources_step_payload_history_selection_and_new_session(self):
        self.context.client = client_for(
            planning_output("both"), response_output("规则与能力。[S1] [R1]"),
            planning_output(), response_output("后续答案。[S1]"),
        )
        app = RAGApp(graph=self.graph, context=self.context, trace_dir=self.root)
        async with app.run_test(size=(140, 40)) as pilot:
            await self.ask(app, pilot, "Warboss 的能力和规则？")
            self.assertEqual(app.stage, "completed")
            self.assertFalse(app.busy)
            self.assertIn("independent", str(app.query_one("#metadata", Static).render()))
            first_run, first_path, session = app.run_id, app.selected_path, app.session_id
            self.assertEqual(read_trace(first_path)["outcome"], "completed")
            app.action_sources("S1")
            await pilot.pause()
            self.assertEqual(json.loads(app.query_one("#source-json", TextArea).text)["sql"], "SELECT name FROM units")
            app.source_ref = ""
            app.show_sources()
            self.assertIn("R1", json.loads(app.query_one("#source-json", TextArea).text))
            app.query_one("#details", TabbedContent).active = "steps"
            table = app.query_one("#timeline", DataTable)
            table.focus()
            await pilot.press("enter")
            payload = json.loads(app.query_one("#payload-json", TextArea).text)
            self.assertEqual(payload[0]["node"], "prepare_turn")
            await self.ask(app, pilot, "它属于哪个阵营？")
            self.assertEqual(app.state["turn_index"], 2)
            self.assertEqual(len(app.events[0]["payload"]["previous_messages"]), 2)
            app.query_one("#runs", Select).value = first_run
            await pilot.pause()
            self.assertEqual(app.run_id, first_run)
            self.assertEqual(app.state["answer"], "规则与能力。[S1] [R1]")
            app.action_new_session()
            await pilot.pause()
            self.assertNotEqual(app.session_id, session)
            self.assertEqual(self.graph.get_state({"configurable": {"thread_id": session}}).values, {})
            self.assertTrue(first_path.exists())
            self.assertEqual(app.runs, {})

    async def test_blocking_model_keeps_ui_responsive_cancel_waits_then_next_turn_works(self):
        entered, release = Event(), Event()
        self.addCleanup(release.set)
        client = client_for(planning_output(), planning_output(), response_output())
        responses = iter(client.models.generate_content.side_effect)

        def generate(**kwargs):
            if not entered.is_set():
                entered.set()
                release.wait(5)
            return next(responses)

        client.models.generate_content.side_effect = generate
        self.context.client = client
        app = RAGApp(graph=self.graph, context=self.context, trace_dir=self.root)
        async with app.run_test(size=(80, 25)) as pilot:
            app.query_one("#question", Input).value = "First question"
            await pilot.press("enter")
            self.assertTrue(await asyncio.to_thread(entered.wait, 5))
            session = app.session_id
            self.assertTrue(app.busy)
            self.assertTrue(app.query_one("#question", Input).disabled)
            self.assertTrue(app.query_one("#new", Button).disabled)
            await pilot.press("ctrl+t")
            self.assertTrue(app.has_class("inspection"))
            self.assertTrue(app.query_one("#inspector").display)
            self.assertFalse(app.query_one("#chat").display)
            await pilot.press("ctrl+n", "escape")
            self.assertEqual(app.session_id, session)
            self.assertTrue(app.busy)
            self.assertTrue(app.cancel_event.is_set())
            self.assertIn("Cancelling", str(app.query_one("#status", Static).render()))
            release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertFalse(app.busy)
            self.assertEqual(app.stage, "cancelled")
            self.assertEqual(app.state["messages"], [])
            self.sql.assert_not_called()
            self.assertEqual(sum(e["event"] == "run.finished" for e in app.events), 1)
            await self.ask(app, pilot, "Second question")
            self.assertEqual(app.stage, "completed")
            self.assertEqual(len(app.state["messages"]), 2)

    async def test_error_and_trace_failure_remain_visible_and_no_trace_mode_writes_nothing(self):
        self.context.client = client_for({})
        self.context.client.models.generate_content.side_effect = RuntimeError("private provider message")
        app = RAGApp(graph=self.graph, context=self.context, trace_dir=self.root)
        async with app.run_test() as pilot:
            with patch("pathlib.Path.mkdir", side_effect=PermissionError("private path")):
                await self.ask(app, pilot, "Question")
            self.assertEqual(app.stage, "error")
            self.assertEqual(app.terminal["failed_node"], "plan")
            self.assertIn("Trace incomplete", str(app.query_one("#notice", Static).render()))
            self.assertNotIn("private provider message", json.dumps(app.events))
            self.assertFalse(app.query_one("#question", Input).disabled)
        self.context.client = client_for(planning_output(), response_output())
        app = RAGApp(graph=self.graph, context=self.context, trace_dir=None)
        async with app.run_test() as pilot:
            await self.ask(app, pilot, "Question")
            self.assertEqual(app.trace_status, "off")
            self.assertIsNone(app.selected_path)
            app.action_sources()
            self.assertIn("S1", json.loads(app.query_one("#source-json", TextArea).text))
        self.assertFalse(self.root.exists())

    async def test_offline_inspector_and_narrow_layout_need_no_graph_or_client(self):
        self.context.client = client_for(planning_output(), response_output())
        result = run_turn(self.graph, self.context, session_id="saved", question="Saved question", trace_dir=self.root)
        trace = read_trace(result["trace_path"])
        app = RAGApp(inspect_path=result["trace_path"], trace=trace)
        async with app.run_test(size=(70, 24)) as pilot:
            self.assertEqual(app.stage, "completed")
            self.assertTrue(app.query_one("#question", Input).disabled)
            self.assertEqual(app.query_one("#timeline", DataTable).row_count, 7)
            await pilot.press("ctrl+s")
            self.assertTrue(app.query_one("#inspector").display)
            self.assertIn("S1", json.loads(app.query_one("#source-json", TextArea).text))
            self.assertGreaterEqual(app.query_one("#source-json", TextArea).content_region.height, 3)
            self.assertEqual(app.query_one("#inspector").region.width, app.query_one("#body").region.width)
            await pilot.resize_terminal(140, 40)
            self.assertFalse(app.has_class("narrow"))
            self.assertTrue(app.query_one("#chat").display)
        with patch("ui.tui.build_agent_graph") as build, patch.object(RAGApp, "run"):
            main(["--inspect", str(result["trace_path"])])
            build.assert_not_called()

    async def test_quit_waits_for_active_call_and_unexpected_runner_errors_unlock_input(self):
        entered, release = Event(), Event()
        self.addCleanup(release.set)
        client = client_for(planning_output())
        response = next(client.models.generate_content.side_effect)

        def generate(**kwargs):
            entered.set()
            release.wait(5)
            return response

        client.models.generate_content.side_effect = generate
        self.context.client = client
        app = RAGApp(graph=self.graph, context=self.context, trace_dir=self.root)
        async with app.run_test() as pilot:
            app.query_one("#question", Input).value = "Question"
            await pilot.press("enter")
            self.assertTrue(await asyncio.to_thread(entered.wait, 5))
            await pilot.press("ctrl+q")
            self.assertTrue(app.quit_when_done)
            self.assertTrue(app.busy)
            self.assertTrue(app.is_running)
            release.set()
            # App exit cancels Textual's worker bookkeeping after the runner's
            # terminal message; wait for that message rather than Worker.wait.
            async with asyncio.timeout(5):
                while app.busy:
                    await asyncio.sleep(0.01)
        self.assertEqual(app.stage, "cancelled")
        self.sql.assert_not_called()
        app = RAGApp(graph=self.graph, context=self.context, trace_dir=None)
        async with app.run_test() as pilot:
            with patch("ui.tui.run_turn", side_effect=OSError("private diagnostic")):
                await self.ask(app, pilot, "Question")
            self.assertFalse(app.busy)
            self.assertEqual(app.stage, "error")
            self.assertIn("Runner failed: OSError", str(app.query_one("#notice", Static).render()))
            self.assertNotIn("private diagnostic", str(app.query_one("#notice", Static).render()))


class CooperativeCancellationTests(unittest.TestCase):
    def test_cancellation_before_commit_skips_following_nodes_and_does_not_commit_answer(self):
        for node in ("plan", "respond", "finalize"):
            with self.subTest(node=node):
                graph = build_agent_graph()
                cancel = Event()
                sql = Mock(return_value=RetrievalResult("ok", "fixed", ["name"], [("Warboss",)]))
                context = EvidenceContext(client=client_for(planning_output(), response_output()), sql_retriever=sql)
                events = []

                def receive(event):
                    events.append(event)
                    if event["event"] == "step.started" and event["node"] == node:
                        cancel.set()

                result = run_turn(graph, context, session_id="cancel", question="Question", trace_dir=None,
                                  on_event=receive, cancel_event=cancel)
                self.assertEqual(result["outcome"], "completed" if node == "finalize" else "cancelled")
                self.assertEqual(len(result["state"]["messages"]), 2 if node == "finalize" else 0)
                self.assertEqual(sum(e["event"] == "run.finished" for e in events), 1)
                if node == "plan":
                    sql.assert_not_called()


if __name__ == "__main__":
    unittest.main()
