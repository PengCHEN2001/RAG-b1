"""Generate the Rulebook benchmark candidates with RAGAS and Gemini.

The final six cases contain three teacher-requested mandatory cases and three
RAGAS-generated candidates. Both JSONL and a human-readable Markdown review
file are written. Generated cases remain pending until manually approved.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data/processed/Core Rules.chunks.jsonl"
DEFAULT_JSONL = ROOT / "evaluation/datasets/rulebook_candidates.jsonl"
DEFAULT_MARKDOWN = ROOT / "evaluation/datasets/rulebook_candidates.md"

MANDATORY_CASES = [
    {
        "case_id": "rb_001",
        "question": "What is the rule for Grenades?",
        "question_type": "single_hop_specific",
        "reference_answer": (
            "In your Shooting phase, select one Grenades unit that is not within "
            "Engagement Range of an enemy and has not been selected to shoot. Select "
            "a visible enemy unit within 8 inches that is not within Engagement Range "
            "of your units, then roll six D6; each 4+ causes 1 mortal wound."
        ),
        "reference_chunk_ids": ["core-rules-grenade-001-001"],
        "origin": "teacher_required",
    },
    {
        "case_id": "rb_002",
        "question": "How can I use a Stratagem?",
        "question_type": "single_hop_specific",
        "reference_answer": (
            "Spend the CP listed on the Stratagem. You cannot use it without enough "
            "CP. A Stratagem may be reused during a battle, but the same Stratagem "
            "cannot normally be used more than once in the same phase."
        ),
        "reference_chunk_ids": ["core-rules-stratagems-001-001"],
        "origin": "teacher_required",
    },
    {
        "case_id": "rb_003",
        "question": "What are the different ways to score points?",
        "question_type": "multi_hop_abstract",
        "reference_answer": (
            "Players score Victory points by achieving the objectives defined by the "
            "mission. Examples include recovering relics, capturing strongholds, "
            "eliminating the opposing Warlord, and controlling objective markers. In "
            "Capture and Control, from battle round two a player scores 1 VP for each "
            "objective marker controlled at the end of their Command phase, up to 3 VP "
            "per turn."
        ),
        "reference_chunk_ids": [
            "core-rules-introduction-001-001",
            "core-rules-read-mission-objectives-001-001",
            "core-rules-capture-and-control-001-001",
        ],
        "origin": "teacher_required",
    },
]

# RAGAS applies several LLM-based transforms to every input document.
# Processing all 212 production chunks would be slow and likely exceed the
# free Gemini rate limit. This fixed,
# representative subset keeps generation reproducible while still using the
# exact chunks retrieved by the application.
RAGAS_SOURCE_CHUNK_IDS = [
    "core-rules-aircraft-in-the-movement-phase-001-001",
    "core-rules-aircraft-and-the-movement-of-other-models-001-001",
    "core-rules-aircraft-in-the-charge-and-fight-phases-001-001",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--model", default=None)
    parser.add_argument("--embedding-model", default=None)
    return parser.parse_args()


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            chunk = json.loads(line)
            if not chunk.get("chunk_id") or not chunk.get("text"):
                raise ValueError(f"Invalid chunk at {path}:{line_number}")
            chunks.append(chunk)
    if not chunks:
        raise ValueError(f"No chunks found in {path}")
    return chunks


def context_record(chunk: dict[str, Any], confidence: float = 100.0) -> dict[str, Any]:
    return {
        "chunk_id": chunk["chunk_id"],
        "text": chunk["text"],
        "source": chunk.get("source"),
        "section": chunk.get("section_path", []),
        "pages": chunk.get("pdf_page_numbers", []),
        "mapping_confidence": round(confidence, 1),
    }


def build_mandatory_cases(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    cases = []
    for seed in MANDATORY_CASES:
        missing = [chunk_id for chunk_id in seed["reference_chunk_ids"] if chunk_id not in by_id]
        if missing:
            raise ValueError(f"Mandatory reference chunks are missing: {missing}")
        cases.append(
            {
                "case_id": seed["case_id"],
                "question": seed["question"],
                "question_type": seed["question_type"],
                "reference_answer": seed["reference_answer"],
                "reference_contexts": [
                    context_record(by_id[chunk_id]) for chunk_id in seed["reference_chunk_ids"]
                ],
                "origin": seed["origin"],
                "review_status": "pending",
            }
        )
    return cases


def select_ragas_source_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    missing = [chunk_id for chunk_id in RAGAS_SOURCE_CHUNK_IDS if chunk_id not in by_id]
    if missing:
        raise ValueError(f"RAGAS source chunks are missing: {missing}")
    return [by_id[chunk_id] for chunk_id in RAGAS_SOURCE_CHUNK_IDS]


def normalize_context(text: str) -> str:
    text = re.sub(r"<\d+-hop>\s*", "", text)
    return " ".join(text.split()).strip().casefold()


def match_context(text: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    from rapidfuzz.fuzz import ratio

    target = normalize_context(text)
    exact = {normalize_context(chunk["text"]): chunk for chunk in chunks}
    if target in exact:
        return context_record(exact[target])

    for normalized, chunk in exact.items():
        if target and (target in normalized or normalized in target):
            confidence = 100.0 * min(len(target), len(normalized)) / max(len(target), len(normalized))
            return context_record(chunk, confidence)

    best_chunk = max(chunks, key=lambda item: ratio(target, normalize_context(item["text"])))
    confidence = ratio(target, normalize_context(best_chunk["text"]))
    return context_record(best_chunk, confidence)


def question_type(synthesizer_name: str) -> str:
    name = synthesizer_name.casefold()
    if "multi_hop_abstract" in name:
        return "multi_hop_abstract"
    if "multi_hop_specific" in name:
        return "multi_hop_specific"
    if "single_hop" in name:
        return "single_hop_specific"
    return "unknown"


def generate_ragas_cases(
    chunks: list[dict[str, Any]], api_key: str, model: str, embedding_model: str
) -> list[dict[str, Any]]:
    from langchain_core.documents import Document
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas.run_config import RunConfig
    from ragas.testset import TestsetGenerator
    from ragas.testset.synthesizers import (
        MultiHopAbstractQuerySynthesizer,
        SingleHopSpecificQuerySynthesizer,
    )

    llm = ChatGoogleGenerativeAI(model=model, api_key=api_key, temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, api_key=api_key)
    generator = TestsetGenerator.from_langchain(llm, embeddings)

    # Two single-hop samples and one theme-based multi-hop sample demonstrate
    # both retrieval modes while keeping the free API usage small. RAGAS's
    # entity-overlap multi-hop-specific strategy is intentionally excluded:
    # closely related rule chunks do not always yield identical named entities.
    distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator.llm), 2 / 3),
        (MultiHopAbstractQuerySynthesizer(llm=generator.llm), 1 / 3),
    ]
    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id": chunk["chunk_id"],
                "source": chunk.get("source"),
                "section_path": chunk.get("section_path", []),
                "pdf_page_numbers": chunk.get("pdf_page_numbers", []),
                "filename": chunk.get("source", "Core Rules.pdf"),
            },
        )
        for chunk in chunks
    ]
    testset = generator.generate_with_langchain_docs(
        documents,
        testset_size=3,
        query_distribution=distribution,
        run_config=RunConfig(
            timeout=120,
            max_retries=3,
            max_wait=10,
            max_workers=2,
            seed=42,
        ),
        raise_exceptions=True,
    )

    rows = testset.to_list()
    if len(rows) != 3:
        raise RuntimeError(f"RAGAS returned {len(rows)} cases; expected 3")

    cases = []
    for index, row in enumerate(rows, start=4):
        contexts = row.get("reference_contexts", [])
        cases.append(
            {
                "case_id": f"rb_{index:03d}",
                "question": row.get("user_input", ""),
                "question_type": question_type(row.get("synthesizer_name", "")),
                "reference_answer": row.get("reference", ""),
                "reference_contexts": [match_context(text, chunks) for text in contexts],
                "origin": "ragas_generated",
                "synthesizer_name": row.get("synthesizer_name"),
                "review_status": "pending",
            }
        )
    return cases


def write_jsonl(path: Path, cases: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")


def markdown_text(cases: list[dict[str, Any]], model: str, embedding_model: str) -> str:
    overall_status = (
        "approved"
        if cases and all(case.get("review_status") == "approved" for case in cases)
        else "pending"
    )
    title = "# Rulebook Benchmark v1" if overall_status == "approved" else "# Rulebook Testset Candidates"
    review_notice = (
        "> All samples have been checked against the canonical project chunks."
        if overall_status == "approved"
        else "> Generated candidates must be manually reviewed before this dataset is frozen as v1."
    )
    lines = [
        title,
        "",
        review_notice,
        "",
        f"- Generator model: `{model}`",
        f"- Embedding model: `{embedding_model}`",
        "- Target distribution: 4 single-hop specific, 2 multi-hop abstract",
        f"- Review status: `{overall_status}`",
        "",
        "| ID | Origin | Type | Question |",
        "|---|---|---|---|",
    ]
    for case in cases:
        question = str(case["question"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {case['case_id']} | {case['origin']} | {case['question_type']} | {question} |"
        )

    for case in cases:
        lines.extend(
            [
                "",
                f"## {case['case_id']} — {case['question_type']}",
                "",
                f"**Question:** {case['question']}",
                "",
                f"**Reference answer:** {case['reference_answer']}",
                "",
                "**Reference chunks:**",
                "",
            ]
        )
        for context in case["reference_contexts"]:
            pages = ", ".join(str(page) for page in context.get("pages", [])) or "unknown"
            sections = " > ".join(context.get("section", [])) or "unknown"
            lines.append(
                f"- `{context['chunk_id']}` — {sections}, page(s) {pages}, "
                f"mapping {context['mapping_confidence']}%"
            )
        review_line = f"**Review:** {case.get('review_status', 'pending')}"
        if case.get("review_notes"):
            review_line += f" — {case['review_notes']}"
        lines.extend(["", review_line, ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key or api_key == "your_google_ai_api_key_here":
        raise SystemExit(
            "GOOGLE_API_KEY is missing. Copy .env.example to .env, add your local key, "
            "and do not commit .env."
        )

    model = args.model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    embedding_model = args.embedding_model or os.getenv(
        "RAGAS_EMBEDDING_MODEL", "models/gemini-embedding-001"
    )
    chunks = load_chunks(args.chunks)
    cases = build_mandatory_cases(chunks)
    ragas_source_chunks = select_ragas_source_chunks(chunks)
    print(
        f"Generating three RAGAS candidates from "
        f"{len(ragas_source_chunks)} representative production chunks..."
    )
    cases.extend(generate_ragas_cases(ragas_source_chunks, api_key, model, embedding_model))

    counts = {
        kind: sum(case["question_type"] == kind for case in cases)
        for kind in ("single_hop_specific", "multi_hop_specific", "multi_hop_abstract")
    }
    expected = {
        "single_hop_specific": 4,
        "multi_hop_specific": 0,
        "multi_hop_abstract": 2,
    }
    if counts != expected:
        raise RuntimeError(f"Unexpected question distribution: {counts}; expected {expected}")

    write_jsonl(args.jsonl, cases)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown_text(cases, model, embedding_model), encoding="utf-8")
    print(f"Wrote {len(cases)} candidates to {args.jsonl}")
    print(f"Wrote review document to {args.markdown}")


if __name__ == "__main__":
    main()
