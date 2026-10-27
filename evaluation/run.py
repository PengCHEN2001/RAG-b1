"""Run the frozen benchmarks and write evaluation results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from dotenv import load_dotenv

from evaluation.adapters import load_evaluation_trace
from evaluation.dataset_loader import (
    BenchmarkDatasets,
    DatasetValidationError,
    load_benchmarks,
)
from evaluation.metrics import (
    aggregate_router_scores,
    aggregate_rulebook_scores,
    aggregate_sql_scores,
    score_rulebook_case,
    score_sql_case,
)
from evaluation.ragas_adapter import evaluate_ragas_cases
from llm.llm_config import get_gemini_model, get_ragas_model


ROOT = Path(__file__).resolve().parent.parent
DATASET_CHOICES = ("all", "rulebook", "text2sql", "combined")
DEFAULT_MODEL = get_gemini_model()
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"


@dataclass(frozen=True)
class OutputPaths:
    raw_results: Path
    summary: Path
    failures: Path
    report: Path
    trace_dir: Path


def _output_paths(root: Path, run_id: str) -> OutputPaths:
    run_dir = root / "runs" / run_id
    report_dir = root / "reports" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)
    return OutputPaths(
        raw_results=run_dir / "raw_results.jsonl",
        summary=report_dir / "summary.json",
        failures=report_dir / "failures.jsonl",
        report=report_dir / "report.md",
        trace_dir=run_dir / "traces",
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _selected_cases(
    datasets: BenchmarkDatasets, selection: str
) -> list[tuple[str, dict[str, Any]]]:
    groups = {
        "rulebook": datasets.rulebook,
        "text2sql": datasets.text2sql,
        "combined": datasets.combined,
    }
    selected = groups if selection == "all" else {selection: groups[selection]}
    return [(name, case) for name, cases in selected.items() for case in cases]


def _expected_route(dataset: str, case: Mapping[str, Any]) -> str:
    if dataset == "rulebook":
        return "rulebook"
    if dataset == "combined":
        return "both"
    return "unsupported" if case.get("expected_status") == "unsupported" else "sql"


def _file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class AgentExecutor:
    """Run each case in an isolated session while reusing loaded resources."""

    def __init__(self, model: str, trace_dir: Path):
        from agent import EvidenceContext, build_agent_graph

        self.graph = build_agent_graph()
        self.context = EvidenceContext(model=model)
        self.trace_dir = trace_dir

    def __call__(self, dataset: str, case: Mapping[str, Any]) -> dict[str, Any]:
        from agent.runtime import run_turn

        result = run_turn(
            self.graph,
            self.context,
            session_id=f"eval-{case['case_id']}-{uuid4().hex[:8]}",
            question=case["question"],
            trace_dir=self.trace_dir,
        )
        if result["trace_path"] is None or not result["trace_complete"]:
            raise RuntimeError("The Agent did not retain a complete trace")
        return load_evaluation_trace(result["trace_path"])


def _score(
    dataset: str,
    case: Mapping[str, Any],
    actual: Mapping[str, Any],
    benchmarks: BenchmarkDatasets,
) -> tuple[dict[str, Any], bool]:
    route_correct = actual.get("route") == _expected_route(dataset, case)
    if dataset == "rulebook":
        score = score_rulebook_case(case, actual.get("rulebook_results", []))
        return score, route_correct and score["recall_at_k"] == 1.0
    if dataset == "text2sql":
        sql_result = actual.get("sql_result")
        if sql_result is None:
            status = "unsupported" if actual.get("route") == "unsupported" else "not_called"
            sql_result = {"status": status}
        score = score_sql_case(case, sql_result, latency_ms=actual.get("duration_ms"))
        return score, route_correct and bool(score["result_correct"])
    sql_cases = {item["case_id"]: item for item in benchmarks.text2sql}
    rulebook_cases = {item["case_id"]: item for item in benchmarks.rulebook}

    sql_result = actual.get("sql_result")
    sql_scores = [
        score_sql_case(sql_cases[case_id], sql_result or {"status": "not_called"})
        for case_id in case["sql_reference_case_ids"]
    ]
    sql_correct = bool(sql_scores) and all(score["result_correct"] for score in sql_scores)

    reference_contexts = [
        context
        for case_id in case["rulebook_reference_case_ids"]
        for context in rulebook_cases[case_id]["reference_contexts"]
    ]
    rulebook_score = score_rulebook_case(
        {"case_id": case["case_id"], "reference_contexts": reference_contexts},
        actual.get("rulebook_results", []),
    )
    rulebook_correct = rulebook_score["recall_at_k"] == 1.0

    required_sources = set(case["required_sources"])
    retrievers_called = set(actual.get("retrievers_called", []))
    citation_sources = set(actual.get("citation_sources", []))
    calls_complete = required_sources.issubset(retrievers_called)
    citations_complete = required_sources.issubset(citation_sources)
    outcome_completed = actual.get("outcome") == "completed"

    score = {
        "route_correct": route_correct,
        "retrievers_called": sorted(retrievers_called),
        "required_retrievers_called": calls_complete,
        "sql_reference_scores": sql_scores,
        "sql_result_correct": sql_correct,
        "rulebook_reference_score": rulebook_score,
        "rulebook_contexts_correct": rulebook_correct,
        "citation_sources": sorted(citation_sources),
        "required_sources_cited": citations_complete,
        "outcome_completed": outcome_completed,
        "answer_review": "manual_required",
    }
    passed = all((
        route_correct,
        calls_complete,
        sql_correct,
        rulebook_correct,
        citations_complete,
        outcome_completed,
    ))
    return score, passed


def _ragas_inputs(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    inputs = []
    for record in records:
        if record["dataset"] != "rulebook" or record["error_type"] is not None:
            continue
        case, actual = record["reference"], record["actual"]
        contexts = []
        for result in actual.get("rulebook_results", []):
            chunk = result.get("chunk", result)
            if isinstance(chunk, Mapping) and isinstance(chunk.get("text"), str):
                contexts.append(chunk["text"])
        inputs.append({
            "case_id": record["case_id"],
            "user_input": record["question"],
            "retrieved_contexts": contexts,
            "reference_contexts": [item["text"] for item in case["reference_contexts"]],
            "response": actual.get("answer"),
            "reference": case["reference_answer"],
        })
    return inputs


def _summary(
    records: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    selection: str,
    model: str,
    judge_model: str | None,
) -> dict[str, Any]:
    valid = [record for record in records if record["error_type"] != "invalid_case"]
    rulebook = [r["deterministic"] for r in valid if r["dataset"] == "rulebook" and r["deterministic"]]
    text2sql = [r["deterministic"] for r in valid if r["dataset"] == "text2sql" and r["deterministic"]]
    router = [
        {"expected_route": r["expected_route"], "predicted_route": r["predicted_route"] or "missing"}
        for r in valid if r.get("expected_route")
    ]
    ragas_names = (
        "context_precision", "context_recall", "faithfulness",
        "response_relevancy", "factual_correctness",
    )
    ragas = {}
    for name in ragas_names:
        values = [r["ragas"][name] for r in valid if r.get("ragas") and r["ragas"].get(name) is not None]
        ragas[name] = sum(values) / len(values) if values else None

    failed = [r for r in records if r["error_type"] or not r["benchmark_passed"]]
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_selection": selection,
        "models": {"agent": model, "ragas_judge": judge_model},
        "versions": {
            "git_commit": _git_commit(),
            "sqlite_sha256": _file_hash(ROOT / "data/sql/wahadb.sqlite"),
            "chunks_sha256": _file_hash(ROOT / "data/processed/Core Rules.chunks.jsonl"),
            "index_manifest_sha256": _file_hash(ROOT / "data/indexes/core_rules/index_manifest.json"),
        },
        "counts": {
            "total": len(records),
            "passed": len(records) - len(failed),
            "failed": len(failed),
            "system_error": sum(r["error_type"] == "system_error" for r in records),
            "evaluation_error": sum(r["error_type"] == "evaluation_error" for r in records),
            "invalid_case": sum(r["error_type"] == "invalid_case" for r in records),
        },
        "deterministic": {
            "rulebook": aggregate_rulebook_scores(rulebook) if rulebook else None,
            "text2sql": aggregate_sql_scores(text2sql) if text2sql else None,
            "router": aggregate_router_scores(router) if router else None,
        },
        "ragas": ragas,
    }


def _report(summary: Mapping[str, Any], failures: Sequence[Mapping[str, Any]]) -> str:
    counts = summary["counts"]
    lines = [
        "# Evaluation Report", "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Dataset: `{summary['dataset_selection']}`",
        f"- Agent model: `{summary['models']['agent']}`",
        f"- RAGAS judge: `{summary['models']['ragas_judge'] or 'not run'}`", "",
        "## Results", "",
        f"- Total: {counts['total']}",
        f"- Passed: {counts['passed']}",
        f"- Failed: {counts['failed']}", "",
        "### Deterministic metrics", "", "```json",
        json.dumps(summary["deterministic"], indent=2), "```", "",
        "### RAGAS metrics", "", "```json",
        json.dumps(summary["ragas"], indent=2), "```", "",
        "## Failures", "",
    ]
    lines.extend(
        f"- `{item['case_id']}`: {item['error_type'] or 'benchmark_mismatch'}"
        for item in failures
    )
    if not failures:
        lines.append("No failures recorded.")
    return "\n".join(lines) + "\n"


def _persist(paths: OutputPaths, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    failures = [r for r in records if r["error_type"] or not r["benchmark_passed"]]
    _write_jsonl(paths.raw_results, records)
    paths.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(paths.failures, failures)
    paths.report.write_text(_report(summary, failures), encoding="utf-8")


def run_evaluation(
    *,
    selection: str,
    output_root: Path,
    run_id: str,
    model: str,
    use_ragas: bool,
    judge_model: str,
    embedding_model: str,
    executor: Callable[[str, Mapping[str, Any]], dict[str, Any]] | None = None,
    ragas_evaluator: Callable[..., dict[str, dict[str, float | None]]] = evaluate_ragas_cases,
    datasets: BenchmarkDatasets | None = None,
) -> OutputPaths:
    """Execute selected cases, score them, and always retain failures."""
    if selection not in DATASET_CHOICES:
        raise ValueError(f"Unknown dataset selection: {selection}")
    paths = _output_paths(output_root, run_id)
    try:
        datasets = datasets or load_benchmarks()
    except DatasetValidationError as error:
        records = [{
            "run_id": run_id, "case_id": "dataset", "dataset": selection,
            "question": None, "expected_route": None, "predicted_route": None,
            "benchmark_passed": False, "error_type": "invalid_case",
            "error": {"type": type(error).__name__, "message": str(error)},
            "deterministic": None, "ragas": None, "actual": None,
        }]
        summary = _summary(
            records, run_id=run_id, selection=selection, model=model, judge_model=None
        )
        _persist(paths, records, summary)
        return paths

    executor = executor or AgentExecutor(model, paths.trace_dir)
    records: list[dict[str, Any]] = []
    for dataset, case in _selected_cases(datasets, selection):
        record = {
            "run_id": run_id, "case_id": case["case_id"], "dataset": dataset,
            "question": case["question"], "expected_route": _expected_route(dataset, case),
            "predicted_route": None, "benchmark_passed": False,
            "error_type": None, "error": None, "deterministic": None,
            "ragas": None, "actual": None, "reference": case,
        }
        try:
            actual = executor(dataset, case)
        except Exception as error:
            record["error_type"] = "system_error"
            record["error"] = {"type": type(error).__name__, "message": str(error)}
        else:
            record["actual"] = actual
            record["predicted_route"] = actual.get("route")
            if actual.get("outcome") == "error":
                record["error_type"] = "system_error"
                record["error"] = actual.get("error")
            else:
                try:
                    record["deterministic"], record["benchmark_passed"] = _score(
                        dataset, case, actual, datasets
                    )
                except Exception as error:
                    record["error_type"] = "evaluation_error"
                    record["error"] = {
                        "type": type(error).__name__, "message": str(error)
                    }
        records.append(record)

    if use_ragas:
        try:
            scores = ragas_evaluator(
                _ragas_inputs(records),
                api_key=os.getenv("GOOGLE_API_KEY", ""),
                judge_model=judge_model,
                embedding_model=embedding_model,
            )
            for record in records:
                record["ragas"] = scores.get(record["case_id"])
        except Exception as error:
            for record in records:
                if record["dataset"] == "rulebook" and record["error_type"] is None:
                    record["error_type"] = "evaluation_error"
                    record["error"] = {"type": type(error).__name__, "message": str(error)}

    for record in records:
        record.pop("reference", None)
    summary = _summary(
        records, run_id=run_id, selection=selection, model=model,
        judge_model=judge_model if use_ragas else None,
    )
    _persist(paths, records, summary)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="all")
    parser.add_argument("--ragas", action="store_true")
    parser.add_argument("--model", default=get_gemini_model())
    parser.add_argument("--judge-model", default=get_ragas_model())
    parser.add_argument("--embedding-model", default=os.getenv("RAGAS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--output-root", type=Path, default=ROOT / "evaluation")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args(argv)
    paths = run_evaluation(
        selection=args.dataset, output_root=args.output_root, run_id=args.run_id,
        model=args.model, use_ragas=args.ragas, judge_model=args.judge_model,
        embedding_model=args.embedding_model,
    )
    for name in ("raw_results", "summary", "failures", "report"):
        print(f"{name}: {getattr(paths, name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
