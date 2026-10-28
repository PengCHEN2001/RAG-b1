"""Textual chat and offline inspector over the shared graph trace stream."""

import argparse
import json
from pathlib import Path
from threading import Event, RLock
from time import perf_counter
from uuid import uuid4

from rich.markdown import Markdown
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import (
    Button, DataTable, Footer, Header, Input, RichLog, Select, Static,
    TabbedContent, TabPane, TextArea,
)
from textual.worker import Worker, WorkerState
from tqdm import tqdm

from agent import EvidenceContext, build_agent_graph
from agent.nodes import DEFAULT_MODEL
from agent.runtime import DEFAULT_TRACE_DIR, read_trace, run_turn, source_details
from llm.instrumentation import format_usage, session_stats, summarize_usage


class TraceEvent(Message):
    def __init__(self, record):
        super().__init__()
        self.record = record


class TurnFinished(Message):
    def __init__(self, result):
        super().__init__()
        self.result = result


class RAGApp(App):
    TITLE = "RAG · rulebook + SQL"
    CSS = """
    Screen { background: $background; overflow: hidden; }
    #toolbar { height: 3; }
    #toolbar Button { min-width: 9; margin-right: 1; }
    #body { height: 1fr; }
    #chat { width: 1fr; border: round $primary; }
    #inspector { width: 1fr; min-width: 42; border: round $secondary; }
    #runs { width: 1fr; margin: 0 1; }
    #metadata { height: auto; max-height: 12; padding: 0 1; }
    #details { height: 1fr; }
    #details ContentSwitcher { height: 1fr; }
    TabPane { padding: 0; height: 1fr; }
    DataTable, TextArea { height: 1fr; }
    #notice { height: auto; max-height: 3; color: $warning; padding: 0 1; }
    #status { height: 1; padding: 0 1; color: $accent; }
    #question { margin: 0 1; width: 1fr; }
    .inspector-hidden #inspector { display: none; }
    .narrow #inspector { width: 1fr; min-width: 0; display: none; }
    .narrow.inspection #chat { display: none; }
    .narrow.inspection #inspector { display: block; }
    .narrow #metadata { max-height: 5; }
    .offline #question { display: none; }
    """
    BINDINGS = [
        ("ctrl+n", "new_session", "New"),
        ("ctrl+t", "toggle_inspector", "Inspector"),
        ("ctrl+s", "sources", "Sources"),
        ("escape", "cancel_turn", "Cancel"),
    ]

    def __init__(self, *, graph=None, context=None, trace_dir=DEFAULT_TRACE_DIR,
                 inspect_path=None, trace=None):
        super().__init__()
        # Textual captures stderr with fileno=-1; tqdm's default process lock
        # cannot start its resource tracker against that stream.
        # ponytail: thread workers only; share a process lock if adding processes.
        tqdm.set_lock(RLock())
        self.graph, self.context = graph, context
        self.trace_dir = trace_dir
        self.inspect_path, self.offline_trace = inspect_path, trace
        self.session_id = uuid4().hex
        self.busy = False
        self.cancel_event = Event()
        self.quit_when_done = False
        self.started = 0.0
        self.stage = "ready"
        self.run_id = None
        self.events = []
        self.state = {}
        self.plan = {}
        self.retrievals = {}
        self.terminal = None
        self.selected_path = None
        self.inspected_span = None
        self.source_ref = ""
        self.trace_status = "off" if trace_dir is None else "ready"
        # Keep only one full run in memory. Older runs are loaded from their
        # trace on selection; the chat log has a bounded rendering buffer.
        self.runs = {}
        self.session_runs = {}
        self.calls = {}
        self.stats = {"llm_calls": None, "usage": None}
        self.instrumentation_complete = True

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="toolbar"):
            yield Button("New", id="new", disabled=self.inspect_path is not None)
            yield Button("Cancel", id="cancel", disabled=True)
            yield Button("Inspector", id="inspect")
            yield Select([], prompt="Select a turn", id="runs")
        with Horizontal(id="body"):
            yield RichLog(id="chat", wrap=True, min_width=1, max_lines=1000)
            with Vertical(id="inspector"):
                yield Static("Ready", id="metadata", markup=False)
                with TabbedContent(id="details"):
                    with TabPane("Steps", id="steps"):
                        yield DataTable(id="timeline", cursor_type="row")
                    with TabPane("Sources", id="sources"):
                        yield TextArea(read_only=True, id="source-json", soft_wrap=False)
                    with TabPane("Payload", id="payload"):
                        yield TextArea(read_only=True, id="payload-json", soft_wrap=False)
        yield Static("", id="notice", markup=False)
        yield Static("Ready · /new /trace /sources [ID] /quit", id="status", markup=False)
        yield Input(placeholder="Ask a question…", id="question", disabled=self.inspect_path is not None)
        yield Footer()

    def on_mount(self):
        table = self.query_one("#timeline", DataTable)
        for label, width in (("Node / event", 23), ("Status", 8), ("ms", 10)):
            table.add_column(label, key=label, width=width)
        self.status_timer = self.set_interval(0.2, self.update_status, pause=True)
        if self.offline_trace is not None:
            self.add_class("offline")
            self.sub_title = "Offline trace"
            self.load_view(self.offline_trace, self.inspect_path)
            self.write_conversation()
            self.query_one("#notice", Static).update("Offline · read only · no model or SQL calls")
        else:
            self.query_one("#chat", RichLog).write(Text("Ask about units or rules. Select a turn to inspect its trace."))
            self.query_one("#question", Input).focus()

    def on_resize(self, event):
        self.set_class(event.size.width < 100, "narrow")

    def update_status(self):
        if self.busy:
            text = f"{'Cancelling · waiting for current call' if self.cancel_event.is_set() else self.stage} · {(perf_counter() - self.started):.1f}s"
        else:
            text = f"{self.stage} · /new /trace /sources [ID] /quit"
        self.query_one("#status", Static).update(text)

    def set_busy(self, busy):
        self.busy = busy
        if busy:
            self.status_timer.resume()
        else:
            self.status_timer.pause()
        self.query_one("#question", Input).disabled = busy or self.inspect_path is not None
        self.query_one("#new", Button).disabled = busy or self.inspect_path is not None
        self.query_one("#runs", Select).disabled = busy
        self.query_one("#cancel", Button).disabled = not busy
        if not busy and self.inspect_path is None:
            self.query_one("#question", Input).focus()
        self.update_status()

    @on(Input.Submitted, "#question")
    def submit(self, event):
        question = event.value.strip()
        if self.busy or not question:
            return
        event.input.clear()
        command, _, argument = question.partition(" ")
        if command == "/new":
            self.action_new_session()
        elif command == "/trace":
            self.action_toggle_inspector()
        elif command == "/sources":
            self.action_sources(argument)
        elif command == "/quit":
            self.action_quit()
        elif command.startswith("/"):
            self.query_one("#notice", Static).update("Commands: /new /trace /sources [ID] /quit")
        else:
            self.reset_view()
            self.cancel_event = Event()
            self.started = perf_counter()
            self.stage = "starting"
            self.query_one("#notice", Static).update("")
            chat = self.query_one("#chat", RichLog)
            chat.write(Text(f"You · {question}", style="bold cyan"))
            self.set_busy(True)
            self.execute_turn(question, self.session_id, self.cancel_event)

    @work(thread=True, exit_on_error=False)
    def execute_turn(self, question, session_id, cancel_event):
        result = run_turn(
            self.graph, self.context, session_id=session_id, question=question,
            trace_dir=self.trace_dir, cancel_event=cancel_event,
            on_event=lambda event: self.post_message(TraceEvent(event)),
        )
        self.post_message(TurnFinished(result))

    def on_trace_event(self, message):
        # New session/next turn remain disabled until the worker returns, so
        # there is no old worker capable of writing into the next run.
        self.consume(message.record)

    def consume(self, event):
        self.events.append(event)
        kind, payload = event["event"], event["payload"]
        table = self.query_one("#timeline", DataTable)
        if kind == "run.started":
            self.run_id = event["run_id"]
            self.state = {"user_input": payload["user_input"], "turn_index": payload["turn_index"]}
            self.trace_status = "recording" if payload["trace_enabled"] else "off"
            self.stats = {"llm_calls": payload.get("llm_calls"), "usage": payload.get("usage")}
        elif kind == "step.started":
            self.stage = event["node"]
            table.add_row(event["node"], "running", "—", key=event["span_id"])
            if not payload.get("llm_instrumentation_complete", True):
                self.instrumentation_complete = False
                self.update_llm_stats()
        elif kind in ("step.finished", "step.failed"):
            key = event["span_id"]
            table.update_cell(key, "Status", "done" if kind == "step.finished" else "failed")
            table.update_cell(key, "ms", f"{payload['duration_ms']:.1f}")
        elif kind.startswith("llm."):
            self.calls[payload["call_id"]] = payload
            key = event["span_id"]
            if kind == "llm.started":
                table.add_row(f"↳ {payload['purpose']}", "running", "—", key=key)
            else:
                table.update_cell(key, "Status", "returned" if kind == "llm.finished" else "failed")
                table.update_cell(key, "ms", f"{payload['duration_ms']:.1f}")
            self.update_llm_stats()
        elif kind in ("route.selected", "query.refined"):
            self.plan = payload["route_plan"]
            self.state.update(payload)
        elif kind == "retrieval.completed":
            self.retrievals[payload["source"]] = payload
        elif kind == "run.finished":
            self.terminal = payload
            self.state = payload["state"]
            self.stage = payload["outcome"]
            self.stats = {"llm_calls": payload.get("llm_calls"), "usage": payload.get("usage")}
            self.session_runs[event["run_id"]] = self.stats
            table.add_row("run.finished", self.stage, f"{payload['duration_ms']:.1f}", key=event["run_id"])
        elif kind == "trace.failed":
            self.trace_status = "INCOMPLETE"
            self.query_one("#notice", Static).update(f"Trace incomplete: {payload['error_type']}. Interaction can continue.")
        self.update_metadata()

    def update_llm_stats(self):
        self.stats = {"llm_calls": len(self.calls), "usage": summarize_usage(
            (call["usage"] for call in self.calls.values()),
            instrumentation_complete=self.instrumentation_complete,
        )}

    def update_metadata(self):
        first = self.events[0] if self.events else None
        if first is None:
            self.query_one("#metadata", Static).update(f"Session {self.session_id[:8]} · ready")
            return
        plan = self.plan
        route = plan.get("route", "—")
        if route == "both":
            route += " · SQL → rules" if plan["depends_on_sql"] else " · independent"
        sql, rules = self.retrievals.get("sql", {}), self.retrievals.get("rulebook", {})
        total = f"{self.terminal['duration_ms']:.1f} ms" if self.terminal else "running"
        failed = self.terminal and self.terminal.get("error")
        status = f"{self.stage} @ {self.terminal['failed_node'] or 'runtime'}" if failed else self.stage
        clipped = sql.get("truncated") or self.state.get("sql_context_truncated") or any(self.state.get("answer_context_truncated", {}).values())
        totals = session_stats({**self.session_runs, self.run_id: self.stats}.values())
        lines = [
            f"Session {first['session_id'][:8]} · run {first['run_id'][:8]} · turn {first['payload']['turn_index']}",
            f"Route {route} · {status} · {total}",
            f"LLM {self.stats['llm_calls']} · {format_usage(self.stats['usage'])}",
            f"Session LLM {totals['llm_calls']} · {format_usage(totals['usage'])}",
            f"SQL {sql.get('strategy', '—')} · rows {sql.get('count', '—')} · chunks {rules.get('count', '—')} · trace {self.trace_status}" + (" · clipped" if clipped else ""),
            f"Model {first['payload']['model']} · history {len(first['payload']['previous_messages'])} messages",
        ]
        if plan.get("reason"):
            lines.append(plan["reason"])
        if failed:
            lines.insert(5, f"{failed['type']} · see Payload")
        self.query_one("#metadata", Static).update("\n".join(lines))

    def on_turn_finished(self, message):
        result = message.result
        self.selected_path = result["trace_path"]
        self.trace_status = "off" if result["trace_complete"] is None else "saved" if result["trace_complete"] else "INCOMPLETE"
        self.remember_run()
        self.write_answer()
        self.update_metadata()
        self.refresh_details()
        self.set_busy(False)
        if self.quit_when_done:
            self.exit()

    @on(Worker.StateChanged)
    def worker_changed(self, event):
        if event.state == WorkerState.ERROR:
            self.stage = "error"
            self.query_one("#notice", Static).update(f"Runner failed: {type(event.worker.error).__name__}. Trace may be incomplete.")
            self.query_one("#chat", RichLog).write(Text("Turn failed before completion.", style="red"))
            self.set_busy(False)
            if self.quit_when_done:
                self.exit()

    def remember_run(self):
        if self.run_id is None:
            return
        number = self.events[0]["payload"]["turn_index"]
        question = self.events[0]["payload"]["user_input"].replace("\n", " ")[:45]
        self.runs[self.run_id] = (f"{number} · {self.stage} · {question}", self.selected_path)
        selector = self.query_one("#runs", Select)
        # Without retained traces only the current run is selectable.
        selector.set_options((Text(label), key) for key, (label, path) in self.runs.items()
                             if path is not None or key == self.run_id)
        selector.value = self.run_id

    @on(Select.Changed, "#runs")
    def select_run(self, event):
        if event.value == Select.NULL or event.value == self.run_id or self.busy:
            return
        path = self.runs[event.value][1]
        try:
            trace = read_trace(path)
        except (OSError, ValueError) as error:
            self.query_one("#notice", Static).update(f"Cannot read trace: {error}")
            return
        self.load_view(trace, path)

    def reset_view(self):
        self.run_id, self.terminal, self.selected_path = None, None, None
        self.inspected_span, self.source_ref = None, ""
        self.events, self.state, self.plan, self.retrievals = [], {}, {}, {}
        self.calls = {}
        self.stats = {"llm_calls": None, "usage": None}
        self.instrumentation_complete = True
        self.query_one("#timeline", DataTable).clear()
        self.query_one("#source-json", TextArea).load_text("")
        self.query_one("#payload-json", TextArea).load_text("")
        self.query_one("#details", TabbedContent).active = "steps"

    def load_view(self, trace, path):
        self.reset_view()
        self.query_one("#notice", Static).update(" · ".join(trace["warnings"]))
        for event in trace["events"]:
            self.consume(event)
        self.stage = trace["outcome"]
        self.selected_path = path
        if self.trace_status != "INCOMPLETE":
            self.trace_status = "saved" if self.terminal else "interrupted"
        self.remember_run()
        self.update_metadata()
        self.update_status()
        self.refresh_details()

    def write_answer(self):
        chat = self.query_one("#chat", RichLog)
        if self.state.get("answer"):
            chat.write(Markdown(self.state["answer"]))
        else:
            chat.write(Text(f"[{self.stage}] No answer committed. See Inspector.", style="yellow"))
        chat.write(Text("─" * 24, style="dim"))

    def write_conversation(self):
        if self.events:
            self.query_one("#chat", RichLog).write(Text(f"You · {self.events[0]['payload']['user_input']}", style="bold cyan"))
            self.write_answer()

    @on(DataTable.RowSelected, "#timeline")
    def inspect_step(self, event):
        self.inspected_span = event.row_key.value
        self.query_one("#details", TabbedContent).active = "payload"
        self.refresh_details()

    @on(TabbedContent.TabActivated, "#details")
    def tab_activated(self):
        self.refresh_details()

    def refresh_details(self):
        active = self.query_one("#details", TabbedContent).active
        if active == "sources":
            self.show_sources()
        elif active == "payload":
            value = [record for record in self.events if record["span_id"] == self.inspected_span] if self.inspected_span else {
                "trace_path": str(self.selected_path) if self.selected_path else None,
                "run": self.terminal, "route_plan": self.plan,
                "session_stats": session_stats(self.session_runs.values()),
                "conversation": self.events[0]["payload"] if self.events else None,
            }
            self.query_one("#payload-json", TextArea).load_text(json.dumps(value, ensure_ascii=False, indent=2))

    def show_sources(self):
        ref = self.source_ref
        sources = source_details(self.state)
        value = sources.get(ref, {"message": f"No cited source {ref}."}) if ref else sources
        self.query_one("#source-json", TextArea).load_text(json.dumps(value, ensure_ascii=False, indent=2))

    def show_inspector(self):
        self.remove_class("inspector-hidden")
        self.add_class("inspection")

    def action_toggle_inspector(self):
        if self.has_class("narrow"):
            self.toggle_class("inspection")
            self.remove_class("inspector-hidden")
        else:
            self.toggle_class("inspector-hidden")

    def action_sources(self, ref=""):
        self.source_ref = ref
        self.show_inspector()
        self.query_one("#details", TabbedContent).active = "sources"
        self.show_sources()

    def action_new_session(self):
        if self.busy or self.inspect_path is not None:
            return
        self.graph.checkpointer.delete_thread(self.session_id)
        self.session_id = uuid4().hex
        self.runs.clear()
        self.session_runs.clear()
        self.reset_view()
        self.query_one("#runs", Select).set_options([])
        self.query_one("#chat", RichLog).clear()
        self.query_one("#notice", Static).update("New session · previous trace files retained")
        self.stage = "ready"
        self.update_metadata()
        self.update_status()

    def action_cancel_turn(self):
        if self.busy:
            self.cancel_event.set()
            self.query_one("#cancel", Button).disabled = True
            self.update_status()

    def action_quit(self):
        if self.busy:
            self.quit_when_done = True
            self.action_cancel_turn()
        else:
            self.exit()

    @on(Button.Pressed)
    def button_pressed(self, event):
        {"new": self.action_new_session, "cancel": self.action_cancel_turn,
         "inspect": self.action_toggle_inspector}[event.button.id]()


def main(argv=None):
    parser = argparse.ArgumentParser(description="RAG chat with a live graph inspector")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--no-trace", action="store_true", help="Disable local trace files")
    parser.add_argument("--inspect", type=Path, help="Open one saved run offline (read only)")
    args = parser.parse_args(argv)
    trace = None
    if args.inspect is not None:
        try:
            trace = read_trace(args.inspect)
        except (OSError, ValueError) as error:
            parser.error(f"Cannot read trace: {error}")
    RAGApp(
        graph=None if trace is not None else build_agent_graph(),
        context=None if trace is not None else EvidenceContext(model=args.model),
        trace_dir=None if args.no_trace else args.trace_dir,
        inspect_path=args.inspect, trace=trace,
    ).run()


if __name__ == "__main__":
    main()
