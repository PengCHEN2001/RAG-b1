"""Offline integration tests for the benchmark runner and report outputs."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.dataset_loader import DatasetValidationError, load_benchmarks
from evaluation.ragas_adapter import evaluate_ragas_cases
from evaluation.run import run_evaluation


class EvaluationRunnerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        self.datasets = load_benchmarks()

    def successful_result(self, dataset, case):
        expected_route = (
            "rulebook" if dataset == "rulebook"
            else "both" if dataset == "combined"
            else "unsupported" if case.get("expected_status") == "unsupported"
            else "sql"
        )
        rulebook_results = []
        sql_result = None
        if dataset == "rulebook":
            rulebook_results = [
                {"chunk": {"chunk_id": context["chunk_id"], "text": context["text"]}}
                for context in case["reference_contexts"]
            ]
        elif dataset == "text2sql" and case["expected_status"] != "unsupported":
            sql_result = {
                "status": case["expected_status"],
                "columns": case["expected_columns"],
                "rows": case["expected_rows"],
            }
        elif dataset == "combined":
            sql_case = next(
                item for item in self.datasets.text2sql
                if item["case_id"] == case["sql_reference_case_ids"][0]
            )
            sql_result = {
                "status": sql_case["expected_status"],
                "columns": sql_case["expected_columns"],
                "rows": sql_case["expected_rows"],
            }
            reference_ids = set(case["rulebook_reference_case_ids"])
            rulebook_results = [
                {"chunk": {"chunk_id": context["chunk_id"], "text": context["text"]}}
                for item in self.datasets.rulebook
                if item["case_id"] in reference_ids
                for context in item["reference_contexts"]
            ]
        combined = dataset == "combined"
        return {
            "route": expected_route,
            "rulebook_results": rulebook_results,
            "sql_result": sql_result,
            "answer": case.get("reference_answer", "Expected status response"),
            "outcome": "unsupported" if expected_route == "unsupported" else "completed",
            "retrievers_called": ["sql", "rulebook"] if combined else [],
            "citation_sources": ["sql", "rulebook"] if combined else [],
            "duration_ms": 10.0,
            "error": None,
        }

    def test_all_datasets_are_scored_and_four_outputs_are_written(self):
        ragas_calls = []

        def fake_ragas(cases, **config):
            ragas_calls.extend(cases)
            return {
                case["case_id"]: {
                    "context_precision": 1.0,
                    "context_recall": 1.0,
                    "faithfulness": 1.0,
                    "response_relevancy": 1.0,
                    "factual_correctness": 1.0,
                }
                for case in cases
            }

        paths = run_evaluation(
            selection="all",
            output_root=self.output,
            run_id="test-run",
            model="test-model",
            use_ragas=True,
            judge_model="judge-model",
            embedding_model="embedding-model",
            executor=self.successful_result,
            ragas_evaluator=fake_ragas,
            datasets=self.datasets,
        )

        for path in (paths.raw_results, paths.summary, paths.failures, paths.report):
            self.assertTrue(path.is_file(), path)
        raw = [json.loads(line) for line in paths.raw_results.read_text().splitlines()]
        summary = json.loads(paths.summary.read_text())
        self.assertEqual(len(raw), 20)
        self.assertEqual(summary["counts"]["passed"], 20)
        self.assertEqual(summary["counts"]["failed"], 0)
        self.assertEqual(summary["deterministic"]["rulebook"]["recall_at_k"], 1.0)
        self.assertEqual(summary["deterministic"]["text2sql"]["execution_accuracy"], 1.0)
        self.assertEqual(summary["deterministic"]["router"]["accuracy"], 1.0)
        self.assertEqual(len(ragas_calls), 6)
        self.assertEqual(paths.failures.read_text(), "")
        self.assertIn("# Evaluation Report", paths.report.read_text())

    def test_ragas_is_opt_in_and_failures_are_not_hidden(self):
        calls = []

        def failing_executor(dataset, case):
            result = self.successful_result(dataset, case)
            result["route"] = "sql"
            return result

        def should_not_run(*args, **kwargs):
            calls.append(True)
            return {}

        paths = run_evaluation(
            selection="rulebook",
            output_root=self.output,
            run_id="failed-run",
            model="test-model",
            use_ragas=False,
            judge_model="judge-model",
            embedding_model="embedding-model",
            executor=failing_executor,
            ragas_evaluator=should_not_run,
            datasets=self.datasets,
        )

        summary = json.loads(paths.summary.read_text())
        failures = [json.loads(line) for line in paths.failures.read_text().splitlines()]
        self.assertFalse(calls)
        self.assertEqual(summary["counts"]["failed"], 6)
        self.assertEqual(len(failures), 6)
        self.assertTrue(all(not record["benchmark_passed"] for record in failures))

    def test_empty_ragas_batch_requires_neither_dependencies_nor_api_key(self):
        self.assertEqual(
            evaluate_ragas_cases(
                [], api_key="", judge_model="unused", embedding_model="unused"
            ),
            {},
        )

    def test_combined_requires_two_successful_retrievers_and_both_citations(self):
        mutations = {
            "wrong_route": lambda result: result.update(route="sql"),
            "sql_error": lambda result: result.update(
                sql_result={"status": "error", "columns": [], "rows": []}
            ),
            "wrong_rulebook_chunk": lambda result: result.update(
                rulebook_results=[{"chunk": {"chunk_id": "wrong", "text": "Wrong"}}]
            ),
            "one_retriever": lambda result: result.update(retrievers_called=["sql"]),
            "one_citation_source": lambda result: result.update(citation_sources=["sql"]),
        }
        for index, (name, mutate) in enumerate(mutations.items(), 1):
            with self.subTest(name=name):
                def failing_executor(dataset, case, mutation=mutate):
                    result = self.successful_result(dataset, case)
                    mutation(result)
                    return result

                paths = run_evaluation(
                    selection="combined",
                    output_root=self.output,
                    run_id=f"combined-failure-{index}",
                    model="test-model",
                    use_ragas=False,
                    judge_model="judge-model",
                    embedding_model="embedding-model",
                    executor=failing_executor,
                    datasets=self.datasets,
                )
                records = [
                    json.loads(line)
                    for line in paths.raw_results.read_text().splitlines()
                ]
                self.assertEqual(len(records), 2)
                self.assertTrue(all(not record["benchmark_passed"] for record in records))

    def test_invalid_dataset_is_retained_as_invalid_case(self):
        with patch(
            "evaluation.run.load_benchmarks",
            side_effect=DatasetValidationError("broken benchmark"),
        ):
            paths = run_evaluation(
                selection="all",
                output_root=self.output,
                run_id="invalid-run",
                model="test-model",
                use_ragas=False,
                judge_model="judge-model",
                embedding_model="embedding-model",
            )

        result = json.loads(paths.raw_results.read_text().strip())
        summary = json.loads(paths.summary.read_text())
        self.assertEqual(result["error_type"], "invalid_case")
        self.assertEqual(summary["counts"]["invalid_case"], 1)
        self.assertEqual(summary["counts"]["failed"], 1)

    def test_scoring_failure_is_an_evaluation_error(self):
        def malformed_result(dataset, case):
            result = self.successful_result(dataset, case)
            result["rulebook_results"] = [{"chunk": {}}]
            return result

        paths = run_evaluation(
            selection="rulebook",
            output_root=self.output,
            run_id="evaluation-error-run",
            model="test-model",
            use_ragas=False,
            judge_model="judge-model",
            embedding_model="embedding-model",
            executor=malformed_result,
            datasets=self.datasets,
        )

        records = [json.loads(line) for line in paths.raw_results.read_text().splitlines()]
        self.assertTrue(all(record["error_type"] == "evaluation_error" for record in records))


if __name__ == "__main__":
    unittest.main()
