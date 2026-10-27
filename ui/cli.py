"""Single-question, interactive and offline trace commands using argparse."""

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

from agent import EvidenceContext, build_agent_graph
from agent.nodes import DEFAULT_MODEL
from agent.runtime import DEFAULT_TRACE_DIR, read_trace, run_turn, source_details
from llm.instrumentation import format_usage, session_stats, summarize_usage


def _progress(event):
    kind, payload = event["event"], event["payload"]
    if kind == "step.started":
        print(f"… {event['node']}", file=sys.stderr, flush=True)
    elif kind in ("step.finished", "step.failed"):
        print(f"{'✓' if kind == 'step.finished' else '✗'} {event['node']} {payload['duration_ms']:.1f} ms",
              file=sys.stderr, flush=True)
    elif kind == "route.selected":
        plan = payload["route_plan"]
        print(f"route={plan['route']} depends_on_sql={plan['depends_on_sql']} · {plan['reason']}", file=sys.stderr)
    elif kind == "retrieval.completed":
        print(f"{payload['source']} status={payload['status']} count={payload['count']} "
              f"strategy={payload.get('strategy', '-')} truncated={payload.get('truncated', False)}", file=sys.stderr)
    elif kind == "llm.started":
        print(f"… LLM {payload['purpose']} model={payload['model']}", file=sys.stderr, flush=True)
    elif kind in ("llm.finished", "llm.failed"):
        print(f"{'✓' if kind == 'llm.finished' else '✗'} LLM {payload['purpose']} "
              f"{payload['duration_ms']:.1f} ms {format_usage(summarize_usage([payload['usage']]))}", file=sys.stderr)


def _show_result(result, show_sources, session):
    state = result["state"]
    if state.get("answer") is not None:
        print(state["answer"])
    if result["error"] is not None:
        print(f"{result['outcome']} at {result['failed_node'] or 'runtime'}: {result['error']['type']}. "
              "Check the trace for stack/validation locations; check credentials/network for model failures.", file=sys.stderr)
    plan = state.get("route_plan")
    strategy = state["sql_result"].strategy if state.get("sql_result") is not None else "none"
    print(f"session={result['session_id']} run={result['run_id']} turn={state.get('turn_index', '?')} "
          f"route={plan['route'] if plan else '-'} sql={strategy} "
          f"outcome={result['outcome']} total={result['duration_ms']:.1f} ms", file=sys.stderr)
    print(f"LLM calls={result['llm_calls']} {format_usage(result['usage'])}", file=sys.stderr)
    print(f"Session LLM calls={session['llm_calls']} {format_usage(session['usage'])}", file=sys.stderr)
    if result["trace_path"] is not None:
        print(f"trace={result['trace_path']} ({'saved' if result['trace_complete'] else 'INCOMPLETE'})", file=sys.stderr)
    if show_sources:
        print(json.dumps(source_details(state), ensure_ascii=False, indent=2))


def main(argv=None, *, graph=None, context=None):
    parser = argparse.ArgumentParser(description="Warhammer 40K RAG agent; omit the question for interactive mode.")
    parser.add_argument("question", nargs="?", help="One quoted question")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--trace", action="store_true", help="Show live graph steps on stderr")
    parser.add_argument("--no-trace", action="store_true", help="Disable local trace files")
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--sources", action="store_true", help="Print cited source details after each answer")
    parser.add_argument("--inspect", type=Path, help="Read one run JSONL offline")
    parser.add_argument("--json", action="store_true", help="With --inspect, print complete trace events")
    args = parser.parse_args(argv)
    if args.inspect is not None:
        if args.question is not None:
            parser.error("--inspect cannot be combined with a question")
        try:
            trace = read_trace(args.inspect)
            if args.json:
                print(json.dumps(trace, ensure_ascii=False, indent=2))
            else:
                if trace["events"]:
                    first = trace["events"][0]
                    print(f"session={first['session_id']} run={first['run_id']} "
                          f"question={json.dumps(first['payload'].get('user_input'), ensure_ascii=False)}")
                for event in trace["events"]:
                    _progress(event)
                terminal = next((event["payload"] for event in trace["events"] if event["event"] == "run.finished"), None)
                print(f"outcome={trace['outcome']}")
                if terminal is not None:
                    print(f"total={terminal['duration_ms']:.1f} ms failed_node={terminal['failed_node'] or '-'}")
                    print(f"LLM calls={terminal.get('llm_calls')} {format_usage(terminal.get('usage'))}")
                    print(terminal["state"].get("answer") or "")
                    if terminal["error"] is not None:
                        print(json.dumps(terminal["error"], ensure_ascii=False, indent=2))
                    if args.sources:
                        print(json.dumps(source_details(terminal["state"]), ensure_ascii=False, indent=2))
            for warning in trace["warnings"]:
                print(warning, file=sys.stderr)
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            print(f"Cannot read trace: {error}", file=sys.stderr)
            return 1
        return 0
    if args.json:
        parser.error("--json requires --inspect")

    graph = graph if graph is not None else build_agent_graph()
    context = context if context is not None else EvidenceContext(model=args.model)
    session_id = uuid4().hex
    show_steps = args.trace
    last_result = None
    runs = {}

    def on_event(event):
        if event["event"] == "trace.failed":
            print(f"WARNING: {event['payload']['message']} ({event['payload']['error_type']})", file=sys.stderr)
        elif show_steps:
            _progress(event)

    def ask(question):
        result = run_turn(graph, context, session_id=session_id, question=question,
                          trace_dir=None if args.no_trace else args.trace_dir, on_event=on_event)
        runs[result["run_id"]] = {"llm_calls": result["llm_calls"], "usage": result["usage"]}
        _show_result(result, args.sources, session_stats(runs.values()))
        return result

    if args.question is not None:
        if not args.question.strip():
            parser.error("question must not be empty")
        result = ask(args.question)
        return 130 if result["outcome"] == "cancelled" else 1 if result["outcome"] == "error" else 0

    print("RAG agent · /new /sources [ID] /trace /quit", file=sys.stderr)
    while True:
        try:
            print("> ", end="", file=sys.stderr, flush=True)
            question = input()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return 0
        if not question.strip():
            continue
        command, _, argument = question.strip().partition(" ")
        if command == "/quit":
            return 0
        if command == "/new":
            graph.checkpointer.delete_thread(session_id)
            session_id, last_result = uuid4().hex, None
            runs.clear()
            print(f"New session: {session_id}", file=sys.stderr)
        elif command == "/trace":
            show_steps = not show_steps
            print(f"Step display {'on' if show_steps else 'off'}; disk trace {'off' if args.no_trace else 'on'}.", file=sys.stderr)
        elif command == "/sources":
            sources = source_details(last_result["state"]) if last_result is not None else {}
            print(json.dumps(sources.get(argument, {}) if argument else sources, ensure_ascii=False, indent=2))
        elif command.startswith("/"):
            print("Commands: /new /sources [ID] /trace /quit", file=sys.stderr)
        else:
            last_result = ask(question)


if __name__ == "__main__":
    raise SystemExit(main())
