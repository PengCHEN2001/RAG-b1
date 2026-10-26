"""Tests for deterministic evaluation metrics."""

import unittest

from evaluation.metrics import (
    MetricInputError,
    aggregate_rulebook_scores,
    aggregate_router_scores,
    aggregate_sql_scores,
    score_rulebook_case,
    score_sql_case,
)


class RulebookMetricTests(unittest.TestCase):
    def test_scores_ranked_reference_chunk_ids(self):
        case = {
            "case_id": "rb_001",
            "reference_contexts": [{"chunk_id": "a"}, {"chunk_id": "b"}],
        }
        retrieved = [
            {"chunk": {"chunk_id": "x"}},
            {"chunk": {"chunk_id": "a"}},
            {"chunk": {"chunk_id": "b"}},
        ]

        score = score_rulebook_case(case, retrieved, k=5)

        self.assertEqual(score["hit_at_k"], 1.0)
        self.assertEqual(score["precision_at_k"], 2 / 5)
        self.assertEqual(score["recall_at_k"], 1.0)
        self.assertEqual(score["first_relevant_rank"], 2)
        self.assertEqual(score["reciprocal_rank"], 0.5)

    def test_accepts_flat_retriever_output_and_reports_misses(self):
        case = {"case_id": "rb_002", "reference_contexts": [{"chunk_id": "wanted"}]}
        score = score_rulebook_case(case, [{"chunk_id": "other"}])
        self.assertEqual(score["hit_at_k"], 0.0)
        self.assertEqual(score["missing_reference_chunk_ids"], ["wanted"])

    def test_aggregates_macro_scores(self):
        summary = aggregate_rulebook_scores([
            {"hit_at_k": 1, "precision_at_k": 0.2, "recall_at_k": 1, "reciprocal_rank": 1},
            {"hit_at_k": 0, "precision_at_k": 0, "recall_at_k": 0, "reciprocal_rank": 0},
        ])
        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["hit_at_k"], 0.5)
        self.assertEqual(summary["mrr"], 0.5)


class SqlMetricTests(unittest.TestCase):
    def test_compares_unordered_rows_numbers_and_repairs_known_mojibake(self):
        case = {
            "case_id": "sql_004",
            "expected_status": "ok",
            "expected_columns": ["ability_name", "value"],
            "expected_rows": [["Martial Ka’tah", "4"], ["Leader", 2]],
            "order_sensitive": False,
        }
        actual = {
            "status": "ok",
            "columns": ["ABILITY_NAME", "value"],
            "rows": [["Leader", "2"], ["Martial Kaâ\x80\x99tah", 4]],
        }

        score = score_sql_case(case, actual, latency_ms=12.5)

        self.assertTrue(score["columns_correct"])
        self.assertTrue(score["rows_correct"])
        self.assertTrue(score["execution_correct"])
        self.assertEqual(score["latency_ms"], 12.5)

    def test_order_sensitive_case_detects_reordered_rows(self):
        case = {
            "case_id": "sql_001",
            "expected_status": "ok",
            "expected_columns": ["name"],
            "expected_rows": [["A"], ["B"]],
            "order_sensitive": True,
        }
        score = score_sql_case(
            case, {"status": "ok", "columns": ["name"], "rows": [["B"], ["A"]]}
        )
        self.assertFalse(score["rows_correct"])
        self.assertFalse(score["execution_correct"])

    def test_empty_and_unsupported_are_scored_by_status(self):
        empty = score_sql_case(
            {"case_id": "sql_010", "expected_status": "empty"},
            {"status": "empty", "rows": []},
        )
        unsupported = score_sql_case(
            {"case_id": "sql_012", "expected_status": "unsupported"},
            {"status": "unsupported"},
        )
        self.assertTrue(empty["result_correct"])
        self.assertTrue(unsupported["result_correct"])
        self.assertIsNone(unsupported["execution_correct"])

    def test_aggregate_uses_only_ok_cases_for_execution_accuracy(self):
        summary = aggregate_sql_scores([
            {
                "execution_correct": True,
                "status_correct": True,
                "result_correct": True,
                "is_error": False,
                "latency_ms": 10,
            },
            {
                "execution_correct": None,
                "status_correct": False,
                "result_correct": False,
                "is_error": True,
                "latency_ms": 30,
            },
        ])
        self.assertEqual(summary["execution_accuracy"], 1.0)
        self.assertEqual(summary["status_accuracy"], 0.5)
        self.assertEqual(summary["error_rate"], 0.5)
        self.assertEqual(summary["mean_latency_ms"], 20.0)

    def test_rejects_rows_that_do_not_match_actual_columns(self):
        case = {
            "case_id": "sql_bad",
            "expected_status": "ok",
            "expected_columns": ["name"],
            "expected_rows": [["A"]],
            "order_sensitive": False,
        }
        with self.assertRaises(MetricInputError):
            score_sql_case(case, {"status": "ok", "columns": [], "rows": [["A"]]})


class RouterMetricTests(unittest.TestCase):
    def test_computes_multiclass_metrics_and_confusion_matrix(self):
        summary = aggregate_router_scores([
            {"expected_route": "rulebook", "predicted_route": "rulebook"},
            {"expected_route": "sql", "predicted_route": "both"},
            {"expected_route": "both", "predicted_route": "both"},
        ])

        self.assertEqual(summary["accuracy"], 2 / 3)
        self.assertEqual(summary["confusion_matrix"]["sql"]["both"], 1)
        self.assertEqual(summary["per_route"]["sql"]["recall"], 0.0)
        self.assertEqual(summary["per_route"]["both"]["precision"], 0.5)


if __name__ == "__main__":
    unittest.main()
