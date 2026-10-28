"""Freeze the reviewed Rulebook candidates as benchmark v1.

This step deliberately reloads every reference context from the canonical
production chunks.  A reviewed answer therefore cannot silently keep stale
text or page metadata from the RAGAS generation run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from llm.llm_config import get_gemini_model

from evaluation.generate_rulebook_testset import (
    DEFAULT_CHUNKS,
    ROOT,
    context_record,
    load_chunks,
    markdown_text,
    write_jsonl,
)


DEFAULT_CANDIDATES = ROOT / "evaluation/datasets/rulebook_candidates.jsonl"
DEFAULT_OUTPUT = ROOT / "evaluation/datasets/rulebook_v1.jsonl"
DEFAULT_MARKDOWN = ROOT / "evaluation/datasets/rulebook_v1.md"


REVIEWED_CASES: dict[str, dict[str, Any]] = {
    "rb_001": {
        "question": "What is the rule for Grenades?",
        "question_type": "single_hop_specific",
        "reference_answer": (
            "The Grenade Stratagem costs 1CP. In your Shooting phase, select one unit "
            "with the Grenades keyword that is not within Engagement Range of any "
            "enemy units and has not been selected to shoot. Select a visible enemy "
            "unit within 8 inches that is not within Engagement Range of any units "
            "from your army, then roll six D6; each 4+ causes 1 mortal wound."
        ),
        "reference_chunk_ids": ["core-rules-grenade-001-001"],
        "review_notes": "Added the 1CP cost after restoring it to the canonical chunk from PDF page 42.",
    },
    "rb_002": {
        "question": "How can I use a Stratagem?",
        "question_type": "single_hop_specific",
        "reference_answer": (
            "Spend the CP listed on the Stratagem. You cannot use it without enough "
            "CP. A Stratagem may be reused during a battle, but the same Stratagem "
            "cannot be used more than once in the same phase."
        ),
        "reference_chunk_ids": ["core-rules-stratagems-001-001"],
        "review_notes": "Removed the unsupported qualifier 'normally'.",
    },
    "rb_003": {
        "question": "What are the different ways to score points?",
        "question_type": "multi_hop_abstract",
        "reference_answer": (
            "Players score Victory points by achieving the objectives defined by the "
            "mission. Depending on the mission, these can include recovering relics, "
            "capturing strongholds, or eliminating the opposing Warlord. For example, "
            "Capture and Control awards 1 VP for each objective marker controlled at "
            "the end of the player's Command phase, starting from the second battle "
            "round, up to 3 VP per turn."
        ),
        "reference_chunk_ids": [
            "core-rules-introduction-001-001",
            "core-rules-read-mission-objectives-001-001",
            "core-rules-capture-and-control-001-001",
        ],
        "review_notes": "Distinguishes general mission-dependent scoring from the Capture and Control example.",
    },
    "rb_004": {
        "question": (
            "During its Normal move, an Aircraft cannot complete its minimum 20-inch "
            "move and is therefore placed into Strategic Reserves. When will that "
            "Aircraft return?"
        ),
        "question_type": "single_hop_specific",
        "reference_answer": (
            "If an Aircraft is placed into Strategic Reserves because it cannot "
            "complete its minimum move, it will always arrive from Strategic Reserves "
            "in your next turn."
        ),
        "reference_chunk_ids": ["core-rules-aircraft-in-the-movement-phase-001-001"],
        "review_notes": (
            "Clarified that the Aircraft entered Strategic Reserves because it could "
            "not complete its minimum move, distinguishing this rule from an Aircraft "
            "that starts the battle in Reserves."
        ),
    },
    "rb_005": {
        "question": (
            "Can my unit still make a Normal or Advance move during the Movement phase "
            "if the only enemy models within its Engagement Range are Aircraft?"
        ),
        "question_type": "single_hop_specific",
        "reference_answer": (
            "Yes. If the only enemy models within Engagement Range of the selected unit "
            "are Aircraft models, that unit can still make a Normal or Advance move."
        ),
        "reference_chunk_ids": ["core-rules-aircraft-and-the-movement-of-other-models-001-001"],
        "review_notes": "Accepted after comparison with the canonical chunk.",
    },
    "rb_006": {
        "question": (
            "An Aircraft cannot complete its minimum move and is placed into Strategic "
            "Reserves. When will it return, and after it returns, which units can charge "
            "it or make melee attacks against it?"
        ),
        "question_type": "multi_hop_abstract",
        "reference_answer": (
            "It returns from Strategic Reserves in your next turn. Only units that can "
            "Fly can charge it or make melee attacks against it. The Aircraft itself "
            "cannot declare a charge; it can fight only while within Engagement Range "
            "of an enemy unit that can Fly, and it can make melee attacks only against "
            "units that can Fly."
        ),
        "reference_chunk_ids": [
            "core-rules-aircraft-in-the-movement-phase-001-001",
            "core-rules-aircraft-in-the-charge-and-fight-phases-001-001",
        ],
        "review_notes": "Uses two supporting chunks; corrected the second chunk's page metadata to page 54.",
    },
}


REQUIRED_EVIDENCE = {
    "rb_001": ["COST: 1CP", "WHEN: Your Shooting phase", "within 8\"", "for each 4+"],
    "rb_002": [
        "reduce your CP total",
        "do not have enough CP",
        "multiple times during a battle",
        "not use the same Stratagem more than once in the same phase",
    ],
    "rb_003": [
        "scoring more Victory points",
        "Each mission will state what its primary objectives are",
        "Starting from the second battle round",
        "scores 1 Victory point",
        "maximum of 3VP per turn",
    ],
    "rb_004": ["will always arrive from Strategic Reserves in your next turn"],
    "rb_005": ["can still make a Normal or Advance move"],
    "rb_006": [
        "will always arrive from Strategic Reserves in your next turn",
        "only units that can Fly can select an Aircraft unit as a target of their charge",
        "Only models that can Fly can make melee attacks against Aircraft units",
        "Aircraft units cannot declare a charge",
    ],
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finalize() -> list[dict[str, Any]]:
    candidates = load_jsonl(DEFAULT_CANDIDATES)
    candidate_by_id = {case["case_id"]: case for case in candidates}
    chunks = load_chunks(DEFAULT_CHUNKS)
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

    if set(candidate_by_id) != set(REVIEWED_CASES):
        raise ValueError("Candidate IDs do not match the reviewed benchmark specification.")

    final_cases: list[dict[str, Any]] = []
    for case_id in sorted(REVIEWED_CASES):
        candidate = candidate_by_id[case_id]
        review = REVIEWED_CASES[case_id]
        missing = [chunk_id for chunk_id in review["reference_chunk_ids"] if chunk_id not in chunk_by_id]
        if missing:
            raise ValueError(f"Missing canonical chunks for {case_id}: {missing}")

        final_case = {
            "case_id": case_id,
            "question": review["question"],
            "question_type": review["question_type"],
            "reference_answer": review["reference_answer"],
            "reference_contexts": [
                context_record(chunk_by_id[chunk_id])
                for chunk_id in review["reference_chunk_ids"]
            ],
            "origin": candidate["origin"],
            "review_status": "approved",
            "review_notes": review["review_notes"],
        }
        if candidate.get("synthesizer_name"):
            final_case["synthesizer_name"] = candidate["synthesizer_name"]

        combined_context = "\n".join(
            context["text"] for context in final_case["reference_contexts"]
        ).casefold()
        missing_evidence = [
            phrase for phrase in REQUIRED_EVIDENCE[case_id]
            if phrase.casefold() not in combined_context
        ]
        if missing_evidence:
            raise ValueError(
                f"Reference contexts for {case_id} do not contain required evidence: "
                f"{missing_evidence}"
            )
        final_cases.append(final_case)
    return final_cases


def main() -> None:
    load_dotenv(ROOT / ".env")
    model = get_gemini_model()
    embedding_model = os.getenv("RAGAS_EMBEDDING_MODEL", "models/gemini-embedding-001")
    cases = finalize()
    write_jsonl(DEFAULT_OUTPUT, cases)
    DEFAULT_MARKDOWN.write_text(
        markdown_text(cases, model, embedding_model), encoding="utf-8"
    )
    print(f"Wrote {len(cases)} approved cases to {DEFAULT_OUTPUT}")
    print(f"Wrote review document to {DEFAULT_MARKDOWN}")


if __name__ == "__main__":
    main()
