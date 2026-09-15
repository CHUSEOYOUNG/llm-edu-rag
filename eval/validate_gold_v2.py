"""Validate the V2 annotation seed and keep pending rows out of evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "eval/gold_v2/plan.json"
DEFAULT_CANDIDATES = ROOT / "eval/gold_v2/candidates.jsonl"
DEFAULT_CHUNKS = ROOT / "data/processed/chunks.jsonl"

ALLOWED_CATEGORIES = {
    "single_fact", "multi_evidence", "table_numeric", "follow_up", "unanswerable",
}
ALLOWED_STATUSES = {"assistant_reviewed_draft", "pending_human_review", "human_reviewed"}
ALLOWED_SPLITS = {"development", "test"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", text.lower())


def validate(plan: dict[str, Any], rows: list[dict[str, Any]], chunk_ids: set[str] | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if len(rows) != plan.get("target_count"):
        errors.append(f"문항 수가 목표와 다릅니다: {len(rows)} != {plan.get('target_count')}")

    ids = [row.get("candidate_id") for row in rows]
    expected_ids = [f"v2q{i:03d}" for i in range(1, len(rows) + 1)]
    if ids != expected_ids:
        errors.append("candidate_id는 v2q001부터 순서대로 중복 없이 이어져야 합니다.")
    source_ids = [row.get("source_id") for row in rows]
    if len(source_ids) != len(set(source_ids)):
        errors.append("source_id가 중복되었습니다.")

    expected_fields = {
        "schema_version", "candidate_id", "source_id", "split", "release_status",
        "primary_category", "expected_status", "question", "conversation_context",
        "standalone_question", "draft_reference_answer", "candidate_evidence_chunk_ids",
        "fact_key", "annotation_status", "evaluation_role", "source_origin", "review_notes",
    }
    normalized_questions: dict[str, list[str]] = defaultdict(list)
    fact_splits: dict[str, set[str]] = defaultdict(set)
    quota_counts: Counter[tuple[str, str]] = Counter()

    for row in rows:
        cid = row.get("candidate_id", "<unknown>")
        missing = expected_fields - set(row)
        extra = set(row) - expected_fields
        if missing or extra:
            errors.append(f"{cid}: 필드 불일치 missing={sorted(missing)} extra={sorted(extra)}")
            continue

        split = row["split"]
        category = row["primary_category"]
        status = row["annotation_status"]
        if split not in ALLOWED_SPLITS:
            errors.append(f"{cid}: 허용되지 않은 split {split}")
        if category not in ALLOWED_CATEGORIES:
            errors.append(f"{cid}: 허용되지 않은 category {category}")
        if status not in ALLOWED_STATUSES:
            errors.append(f"{cid}: 허용되지 않은 annotation_status {status}")
        quota_counts[(category, split)] += 1

        if split == "test":
            if row["release_status"] != "sealed":
                errors.append(f"{cid}: test 문항은 sealed여야 합니다.")
            if status == "pending_human_review" and row["evaluation_role"] != "withheld_final_test_candidate":
                errors.append(f"{cid}: 검수 전 test 문항은 평가 대상이 될 수 없습니다.")
            if row["source_origin"] == "existing_evidence_paraphrase":
                errors.append(f"{cid}: 기존 근거의 단순 말바꾸기는 test에 넣을 수 없습니다.")
        elif row["release_status"] != "open":
            errors.append(f"{cid}: development 문항은 open이어야 합니다.")

        if status == "pending_human_review" and row["evaluation_role"] not in {
            "development_candidate", "withheld_final_test_candidate"
        }:
            errors.append(f"{cid}: pending 문항이 평가 가능 역할을 가집니다.")
        if status == "assistant_reviewed_draft" and (
            split != "development" or row["evaluation_role"] != "development_regression_only"
        ):
            errors.append(f"{cid}: 기존 초안은 development regression 전용이어야 합니다.")

        answerable = row["expected_status"] == "answerable"
        if row["expected_status"] not in {"answerable", "unanswerable"}:
            errors.append(f"{cid}: expected_status가 잘못되었습니다.")
        if answerable:
            if not row["draft_reference_answer"] or not row["candidate_evidence_chunk_ids"]:
                errors.append(f"{cid}: answerable 문항에 답 또는 근거가 없습니다.")
        elif row["draft_reference_answer"] is not None or row["candidate_evidence_chunk_ids"]:
            errors.append(f"{cid}: unanswerable 문항에는 답이나 근거를 두지 않습니다.")

        context = row["conversation_context"]
        if category == "follow_up":
            if len(context) != 2 or [turn.get("role") for turn in context] != ["user", "assistant"]:
                errors.append(f"{cid}: follow_up에는 user/assistant 1턴 문맥이 필요합니다.")
            if normalize(row["standalone_question"]) == normalize(row["question"]):
                errors.append(f"{cid}: follow_up 독립 질문이 현재 발화와 같아서는 안 됩니다.")
        elif context or row["standalone_question"] != row["question"]:
            errors.append(f"{cid}: follow_up이 아닌 문항에는 대화 문맥이나 재작성 질문을 두지 않습니다.")

        if not row["fact_key"].strip():
            errors.append(f"{cid}: fact_key가 없습니다.")
        fact_splits[row["fact_key"]].add(split)
        normalized_questions[normalize(row["standalone_question"])].append(cid)

        if chunk_ids is not None:
            missing_chunks = set(row["candidate_evidence_chunk_ids"]) - chunk_ids
            if missing_chunks:
                errors.append(f"{cid}: 코퍼스에 없는 근거 {sorted(missing_chunks)}")

    for category, splits in plan["category_quotas"].items():
        for split, target in splits.items():
            actual = quota_counts[(category, split)]
            if actual != target:
                errors.append(f"배분 불일치 {category}/{split}: {actual} != {target}")

    for fact_key, splits in fact_splits.items():
        if len(splits) > 1:
            errors.append(f"동일 fact_key가 development와 test에 함께 있습니다: {fact_key}")
    for _, duplicate_ids in normalized_questions.items():
        if len(duplicate_ids) > 1:
            errors.append(f"독립 질문이 중복되었습니다: {duplicate_ids}")

    pending = sum(row["annotation_status"] == "pending_human_review" for row in rows)
    human_reviewed = sum(row["annotation_status"] == "human_reviewed" for row in rows)
    if human_reviewed == 0:
        warnings.append("전문가 또는 사람 검수를 마친 문항이 아직 없습니다. V2 공식 점수를 보고하지 마세요.")

    summary = {
        "valid": not errors,
        "question_count": len(rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "category_split_counts": {
            f"{category}/{split}": quota_counts[(category, split)]
            for category in sorted(ALLOWED_CATEGORIES) for split in sorted(ALLOWED_SPLITS)
        },
        "annotation_counts": dict(sorted(Counter(row["annotation_status"] for row in rows).items())),
        "pending_human_review": pending,
        "human_reviewed": human_reviewed,
        "errors": errors,
        "warnings": warnings,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--skip-chunk-check", action="store_true")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    rows = read_jsonl(args.candidates)
    chunk_ids = None
    if not args.skip_chunk_check and args.chunks.exists():
        chunk_ids = {row["chunk_id"] for row in read_jsonl(args.chunks)}
    summary = validate(plan, rows, chunk_ids)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
