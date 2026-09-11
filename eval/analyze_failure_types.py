"""Aggregate retrieval and generation regression metrics by question structure."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
LAYOUTS = {"single_passage", "multi_passage", "none"}
FAMILIES = {
    "policy_lookup", "conditional_exception", "format_lookup", "list_lookup",
    "cross_scope_comparison", "temporal_multi_hop", "table_lookup", "corpus_abstention",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_taxonomy(taxonomy: dict, questions: list[dict]) -> dict[str, dict]:
    if set(taxonomy) != {"schema_version", "status", "description", "rows"}:
        raise ValueError("분류표 필드를 확인하세요.")
    rows = taxonomy["rows"]
    by_qid = {row["qid"]: row for row in rows}
    if len(by_qid) != len(rows) or set(by_qid) != {q["qid"] for q in questions}:
        raise ValueError("분류표는 평가 질문을 정확히 한 번씩 포함해야 합니다.")
    for row in rows:
        if (set(row) != {"qid", "family", "evidence_layout", "operations", "rationale"}
                or row["family"] not in FAMILIES
                or row["evidence_layout"] not in LAYOUTS
                or not row["operations"]
                or len(row["operations"]) != len(set(row["operations"]))
                or not row["rationale"].strip()):
            raise ValueError(f"분류값을 확인하세요: {row.get('qid')}")
    return by_qid


def mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def summarize(qids: set[str], retrieval: dict[str, dict], generation: dict[str, dict]) -> dict:
    retrieval_rows = [retrieval[qid] for qid in qids if qid in retrieval]
    generation_rows = [generation[qid] for qid in qids if qid in generation]
    answerable_rows = [row for row in generation_rows if row["answerable"]]
    unanswerable_rows = [row for row in generation_rows if not row["answerable"]]
    retrieval_metrics = {}
    if retrieval_rows:
        keys = retrieval_rows[0]["metrics"]
        retrieval_metrics = {
            key: mean([row["metrics"][key] for row in retrieval_rows]) for key in keys
        }
    return {
        "question_count": len(qids),
        "qids": sorted(qids),
        "retrieval": {"n": len(retrieval_rows), "metrics": retrieval_metrics},
        "generation": {
            "n": len(generation_rows),
            "task_success_rate": mean([float(row["task_success"]) for row in generation_rows]),
            "answer_success_rate": mean([
                float(row["actual_status"] == "draft_answer") for row in answerable_rows
            ]),
            "citation_valid_rate": mean([
                float(row["citation_ok"]) for row in answerable_rows
            ]),
            "unanswerable_rejection_rate": mean([
                float(row["actual_status"] == "insufficient_evidence")
                for row in unanswerable_rows
            ]),
            "failed_qids": sorted(row["qid"] for row in generation_rows
                                  if not row["task_success"]),
        },
    }


def run(taxonomy: dict, questions: list[dict], regression: dict) -> dict:
    by_qid = validate_taxonomy(taxonomy, questions)
    retrieval = {row["qid"]: row for row in regression["retrieval"]["per_question"]}
    generation = {row["qid"]: row for row in regression["generation"]["per_question"]}
    if set(generation) != set(by_qid) or not set(retrieval) <= set(by_qid):
        raise ValueError("회귀 결과와 질문 분류표의 범위가 다릅니다.")

    dimensions = {}
    for dimension in ("family", "evidence_layout"):
        groups = defaultdict(set)
        for qid, row in by_qid.items():
            groups[row[dimension]].add(qid)
        dimensions[dimension] = {
            label: summarize(qids, retrieval, generation)
            for label, qids in sorted(groups.items())
        }
    operations = defaultdict(set)
    for qid, row in by_qid.items():
        for operation in row["operations"]:
            operations[operation].add(qid)
    dimensions["operation"] = {
        label: summarize(qids, retrieval, generation)
        for label, qids in sorted(operations.items())
    }
    return {
        "analysis": "question_failure_type_breakdown",
        "status": taxonomy["status"],
        "question_count": len(by_qid),
        "regression_schema_version": regression["schema_version"],
        "dimensions": dimensions,
        "limitations": [
            "Question categories were drafted after observing development failures.",
            "Operation groups overlap and must not be summed as disjoint samples.",
            "Groups with one or two questions are descriptive cases, not stable estimates.",
            "Citation validity checks provenance format, not semantic entailment.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression", type=Path,
                        default=ROOT / "experiments/regression_generation_school_comparison.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/failure_type_breakdown.json")
    args = parser.parse_args()
    taxonomy_path = ROOT / "eval/question_taxonomy.v1.json"
    questions_path = ROOT / "eval/questions.v2.draft.jsonl"
    taxonomy = json.loads(taxonomy_path.read_text())
    questions = read_jsonl(questions_path)
    regression = json.loads(args.regression.read_text())
    result = run(taxonomy, questions, regression)
    result["inputs"] = {
        "taxonomy_sha256": sha256(taxonomy_path),
        "questions_sha256": sha256(questions_path),
        "regression_sha256": sha256(args.regression),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("evidence layout")
    for label, row in result["dimensions"]["evidence_layout"].items():
        r = row["retrieval"]["metrics"]
        g = row["generation"]
        print(f"{label:15} n={row['question_count']:2d} "
              f"Complete@5={r.get('complete@5', 0):.3f} "
              f"answer_success={g['answer_success_rate']}")
    print(f"결과: {args.output}")


if __name__ == "__main__":
    main()
