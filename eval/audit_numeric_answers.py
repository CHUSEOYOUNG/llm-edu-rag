"""Recheck saved generation answers against numbers in draft reference answers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regression_eval import read_jsonl, reference_numbers, reference_numbers_match  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(questions: list[dict], result: dict) -> dict:
    by_qid = {row["qid"]: row for row in questions}
    rows = []
    for generated in result["generation"]["per_question"]:
        question = by_qid[generated["qid"]]
        expected = reference_numbers(question.get("answer"))
        if not generated["answerable"] or not expected:
            continue
        rows.append({
            "qid": generated["qid"],
            "expected_numbers": sorted(expected),
            "answer": generated.get("answer"),
            "match": reference_numbers_match(question.get("answer"), generated.get("answer")),
        })
    matched = sum(row["match"] for row in rows)
    return {
        "analysis": "retrospective_reference_number_match",
        "evaluated_count": len(rows),
        "matched_count": matched,
        "reference_number_match_rate": round(matched / len(rows), 4) if rows else None,
        "per_question": rows,
        "limitation": "Number coverage does not verify units, relationships, or non-numeric meaning.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path,
                        default=ROOT / "eval/gold_v2/development.assistant-reviewed.jsonl")
    parser.add_argument("--result", type=Path,
                        default=ROOT / "experiments/gold_v2_development_generation.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/gold_v2_development_generation_numeric_audit.json")
    args = parser.parse_args()
    audited = audit(read_jsonl(args.questions), json.loads(args.result.read_text()))
    audited["inputs"] = {
        "questions_sha256": sha256(args.questions),
        "result_sha256": sha256(args.result),
    }
    args.output.write_text(json.dumps(audited, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: audited[key] for key in (
        "evaluated_count", "matched_count", "reference_number_match_rate"
    )}, ensure_ascii=False))
    print(f"결과: {args.output}")


if __name__ == "__main__":
    main()
