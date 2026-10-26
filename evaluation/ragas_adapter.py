"""Optional RAGAS judge integration for Rulebook and combined answers.

Imports are intentionally lazy so deterministic evaluation works without the
``eval`` dependency group and never makes an accidental provider request.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


class RagasEvaluationError(RuntimeError):
    """Raised when RAGAS cannot evaluate an otherwise completed system run."""


_OUTPUT_NAMES = {
    "llm_context_precision_with_reference": "context_precision",
    "context_recall": "context_recall",
    "faithfulness": "faithfulness",
    "answer_relevancy": "response_relevancy",
    "factual_correctness": "factual_correctness",
}


def _score_value(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def evaluate_ragas_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    api_key: str,
    judge_model: str,
    embedding_model: str,
) -> dict[str, dict[str, float | None]]:
    """Run five RAGAS metrics and return scores keyed by benchmark case ID."""
    if not cases:
        return {}
    if not api_key:
        raise RagasEvaluationError("GOOGLE_API_KEY is required when --ragas is enabled")

    try:
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
            GoogleGenerativeAIEmbeddings,
        )
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            FactualCorrectness,
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError as error:
        raise RagasEvaluationError(
            "Install the eval dependencies with: uv sync --group eval"
        ) from error

    samples = []
    case_ids = []
    for case in cases:
        case_id = case.get("case_id")
        required = {
            "user_input": case.get("user_input"),
            "retrieved_contexts": case.get("retrieved_contexts"),
            "response": case.get("response"),
            "reference": case.get("reference"),
        }
        if not isinstance(case_id, str) or any(
            value is None or value == "" or value == [] for value in required.values()
        ):
            raise RagasEvaluationError(
                f"RAGAS case {case_id or '<unknown>'} is missing a required field"
            )
        samples.append(SingleTurnSample(
            user_input=required["user_input"],
            retrieved_contexts=list(required["retrieved_contexts"]),
            reference_contexts=list(case.get("reference_contexts", [])),
            response=required["response"],
            reference=required["reference"],
        ))
        case_ids.append(case_id)

    llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model=judge_model,
        api_key=api_key,
        temperature=0,
        retries=2,
    ))
    embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
        model=embedding_model,
        api_key=api_key,
    ))
    metrics = [
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings),
        FactualCorrectness(llm=llm),
    ]
    try:
        result = evaluate(
            EvaluationDataset(samples=samples),
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
            show_progress=True,
        )
        rows = result.to_pandas().to_dict(orient="records")
    except Exception as error:
        raise RagasEvaluationError(f"RAGAS execution failed: {type(error).__name__}") from error
    if len(rows) != len(case_ids):
        raise RagasEvaluationError("RAGAS returned a different number of rows than requested")

    return {
        case_id: {
            public_name: _score_value(row.get(result_name))
            for result_name, public_name in _OUTPUT_NAMES.items()
        }
        for case_id, row in zip(case_ids, rows, strict=True)
    }
