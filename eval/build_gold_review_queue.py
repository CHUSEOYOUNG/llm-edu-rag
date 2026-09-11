"""Prepare existing benchmark questions for human gold-set annotation.

This script never promotes a candidate into the scored gold set.  It only makes
missing review work explicit so reference answers and evidence groups can be
checked before inclusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]


def normalize_question(value: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", value)).lower()


def evidence_ids(question: dict) -> set[str]:
    reviewed = {
        item["chunk_id"]
        for group in question.get("evidence_groups", [])
        for item in group.get("alternatives", [])
    }
    return reviewed | set(question.get("gold_chunks", []))


def build_queue(current: list[dict], candidates: list[dict]) -> list[dict]:
    existing_questions = {normalize_question(row["question"]) for row in current}
    existing_evidence = set().union(*(evidence_ids(row) for row in current))
    queue = []
    for row in candidates:
        if normalize_question(row["question"]) in existing_questions:
            continue
        candidate_evidence = set(row["evidence_chunk_ids"])
        if not row["answerable"]:
            candidate_kind = "unanswerable_candidate"
        elif candidate_evidence & existing_evidence:
            candidate_kind = "existing_evidence_paraphrase"
        else:
            candidate_kind = "new_evidence_candidate"
        queue.append({
            "candidate_id": row["qid"],
            "question": row["question"],
            "expected_status": "answerable" if row["answerable"] else "unanswerable",
            "candidate_kind": candidate_kind,
            "source_benchmark_split": row["split"],
            "source_difficulty": row["difficulty"],
            "candidate_evidence_chunk_ids": row["evidence_chunk_ids"],
            "reference_answer": None,
            "evidence_groups": [],
            "annotation_status": "pending_human_review",
            "review_checks": {
                "question_is_realistic": None,
                "reference_answer_matches_source": None,
                "all_required_facts_grouped": None,
                "school_grade_date_scope_checked": None,
                "unanswerable_verified_against_corpus": None,
            },
            "evaluation_role": "annotation_candidate_only",
        })
    return queue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "eval/gold_expansion_review.jsonl")
    args = parser.parse_args()
    current = [json.loads(line) for line in
               (ROOT / "eval/questions.v2.draft.jsonl").read_text().splitlines()
               if line.strip()]
    candidates = [json.loads(line) for line in
                  (ROOT / "eval/answerability.v2.jsonl").read_text().splitlines()
                  if line.strip()]
    queue = build_queue(current, candidates)
    args.output.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in queue
    ))
    counts = {kind: sum(row["candidate_kind"] == kind for row in queue)
              for kind in ("new_evidence_candidate", "existing_evidence_paraphrase",
                           "unanswerable_candidate")}
    print(f"검수 후보 {len(queue)}개: {json.dumps(counts, ensure_ascii=False)}")
    print(f"결과: {args.output}")


if __name__ == "__main__":
    main()
