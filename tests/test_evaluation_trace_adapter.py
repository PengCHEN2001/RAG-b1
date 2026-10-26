"""Offline tests for the evaluation trace adapter."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.adapters.trace_adapter import TraceAdapterError, load_evaluation_trace


class EvaluationTraceAdapterTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "run.jsonl"

    def write_events(self, events):
        self.path.write_text(
            "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
        )

    def event(self, sequence, event, payload):
        return {
            "schema_version": 1,
            "session_id": "session-1",
            "run_id": "run-1",
            "seq": sequence,
            "timestamp": "2026-10-26T00:00:00+00:00",
            "event": event,
            "node": None,
            "span_id": "run-1",
            "parent_span_id": None,
            "payload": payload,
        }

    def test_extracts_terminal_evaluation_fields(self):
        state = {
            "user_input": "Question",
            "standalone_question": "Question",
            "route_plan": {"route": "both"},
            "sql_result": {"status": "ok", "columns": ["value"], "rows": [["2"]]},
            "rulebook_result": {
                "results": [{"chunk": {"chunk_id": "chunk-1", "text": "Rule"}}]
            },
            "evidence_status": "ok",
            "answer": "Answer [S1] [R1]",
            "citations": {
                "S1": {"source": "sql", "source_id": "run-1:sql"},
                "R1": {"source": "rulebook", "source_id": "chunk-1"},
            },
        }
        self.write_events([
            self.event(1, "run.started", {"user_input": "Question"}),
            self.event(2, "run.finished", {
                "outcome": "completed",
                "duration_ms": 125.5,
                "step_durations_ms": {"plan": 10.0, "respond": 20.0},
                "failed_node": None,
                "error": None,
                "state": state,
                "llm_calls": None,
                "usage": None,
                "first_answer_token_ms": None,
            }),
        ])

        record = load_evaluation_trace(self.path)

        self.assertEqual(record["route"], "both")
        self.assertEqual(record["sql_result"]["rows"], [["2"]])
        self.assertEqual(record["rulebook_results"][0]["chunk"]["chunk_id"], "chunk-1")
        self.assertEqual(record["answer"], "Answer [S1] [R1]")
        self.assertEqual(record["duration_ms"], 125.5)
        self.assertIsNone(record["usage"])

    def test_rejects_an_incomplete_trace(self):
        self.write_events([self.event(1, "run.started", {"user_input": "Question"})])

        with self.assertRaisesRegex(TraceAdapterError, "Incomplete trace"):
            load_evaluation_trace(self.path)


if __name__ == "__main__":
    unittest.main()
