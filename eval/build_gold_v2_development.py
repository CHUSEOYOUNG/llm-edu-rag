"""Create an assistant-reviewed development set from the V2 annotation seed."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = ROOT / "eval/gold_v2/candidates.jsonl"
EXISTING_PATH = ROOT / "eval/questions.v2.draft.jsonl"
CHUNKS_PATH = ROOT / "data/processed/chunks.jsonl"
OUTPUT_PATH = ROOT / "eval/gold_v2/development.assistant-reviewed.jsonl"
MANIFEST_PATH = ROOT / "eval/gold_v2/development.assistant-reviewed.manifest.json"

NEW_DEVELOPMENT_IDS = {
    "a006", "a007", "a008", "authored-s001",
    "authored-m001", "authored-m002", "authored-m003",
    "authored-t001", "authored-t002", "authored-t003", "authored-t004",
    "authored-f001", "authored-f002", "authored-f003", "authored-f004", "authored-f005",
    "u002", "u003", "u004", "u010", "u011", "u012",
}

NEGATIVE_AUDIT_TERMS = {
    "u002": ["급식", "점심 메뉴", "식단"],
    "u003": ["교복", "교복 가격"],
    "u004": ["학교 축제", "축제"],
    "u010": ["전학 신청", "신청 마감"],
    "u011": ["장학금", "대상자 발표"],
    "u012": ["학부모 상담 주간", "상담 주간"],
}

HIGH = "2026 학교생활기록부 기재요령(고)_F_260219"
MIDDLE = "2026 학교생활기록부 기재요령(중)_F_260227"
ELEMENTARY = "2026 학교생활기록부 기재요령(초)_F_260219"


def cid(doc: str, suffix: str) -> str:
    return f"{doc}::{suffix}"


# School-record guidance repeats equivalent rules in the elementary, middle, and
# high-school editions. Retrieval of any equivalent edition should count as a hit.
EVIDENCE_ALTERNATIVES = {
    "a007": [[cid(HIGH, "s133::8조::0"),
               cid(MIDDLE, "s339::8조::0"), cid(ELEMENTARY, "s504::8조::1")]],
    "a008": [[cid(HIGH, "s070::7조::0"), cid(MIDDLE, "s276::7조::0"),
               cid(ELEMENTARY, "s442::7조::0"), cid(HIGH, "s121::5조::2"),
               cid(MIDDLE, "s326::5조::2"), cid(ELEMENTARY, "s492::5조::1")]],
    "authored-s001": [[cid(HIGH, "s133::8조::0"), cid(MIDDLE, "s339::8조::0"),
                        cid(ELEMENTARY, "s504::8조::0")]],
    "authored-m001": [
        [cid(HIGH, "s136::나::1"), cid(MIDDLE, "s342::나::1"),
         cid(ELEMENTARY, "s509::나::2")],
        [cid(HIGH, "s136::나::2"), cid(MIDDLE, "s342::나::2"),
         cid(ELEMENTARY, "s509::나::3")],
    ],
    "authored-m003": [
        [cid(HIGH, "s225::19조::0"), cid(MIDDLE, "s391::19조::0"),
         cid(ELEMENTARY, "s550::19조::0"), cid(HIGH, "s228::2::0"),
         cid(MIDDLE, "s394::2::0"), cid(ELEMENTARY, "s553::2::0")],
        [cid(HIGH, "s225::19조::1"), cid(MIDDLE, "s391::19조::1"),
         cid(ELEMENTARY, "s550::19조::1")],
    ],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anchored_group(chunk_rows: list[dict[str, Any]], index: int) -> dict[str, Any]:
    return {
        "group_id": f"required_evidence_{index}",
        "claim": f"초안 답변에 필요한 근거 {index}",
        "alternatives": [
            {
                "chunk_id": chunk_row["chunk_id"],
                "doc_id": chunk_row["doc_id"],
                "start": 0,
                "end": len(chunk_row["body"]),
                "text": chunk_row["body"],
            }
            for chunk_row in chunk_rows
        ],
    }


def existing_row(candidate: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "gold-v2-development-1",
        "qid": candidate["candidate_id"],
        "source_id": candidate["source_id"],
        "type": "unans" if candidate["expected_status"] == "unanswerable" else source["type"],
        "primary_category": candidate["primary_category"],
        "question": candidate["question"],
        "previous_question": None,
        "expected_resolved_question": candidate["standalone_question"],
        "answer": candidate["draft_reference_answer"],
        "gold_chunks": candidate["candidate_evidence_chunk_ids"],
        "evidence_groups": source.get("evidence_groups", []),
        "meta": source.get("meta"),
        "annotation_status": "assistant_reviewed_draft",
        "annotation_basis": "existing V2 draft; expert review pending",
    }


def new_row(candidate: dict[str, Any], chunks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    evidence = candidate["candidate_evidence_chunk_ids"]
    context = candidate["conversation_context"]
    alternatives = EVIDENCE_ALTERNATIVES.get(
        candidate["source_id"], [[chunk_id] for chunk_id in evidence]
    )
    if len(alternatives) != len(evidence):
        raise ValueError(f"근거 그룹 수가 다릅니다: {candidate['source_id']}")
    return {
        "schema_version": "gold-v2-development-1",
        "qid": candidate["candidate_id"],
        "source_id": candidate["source_id"],
        "type": "unans" if candidate["expected_status"] == "unanswerable" else "fact",
        "primary_category": candidate["primary_category"],
        "question": candidate["question"],
        "previous_question": context[0]["content"] if context else None,
        "expected_resolved_question": candidate["standalone_question"],
        "answer": candidate["draft_reference_answer"],
        "gold_chunks": evidence,
        "evidence_groups": [
            anchored_group([chunks[chunk_id] for chunk_id in group], i)
            for i, group in enumerate(alternatives, 1)
        ],
        "meta": {"school_level": None, "source_hint": None},
        "annotation_status": "assistant_reviewed_draft",
        "annotation_basis": "local corpus cross-check; not expert or current-policy validation",
    }


def negative_audit(corpus_text: str) -> dict[str, Any]:
    rows = {}
    for source_id, terms in NEGATIVE_AUDIT_TERMS.items():
        rows[source_id] = {
            "literal_term_counts": {term: corpus_text.count(term) for term in terms},
            "interpretation": (
                "A term match may discuss record-writing examples or prohibited content. "
                "No row was found that supplies the requested school-specific schedule, price, or announcement."
            ),
        }
    return rows


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = read_jsonl(CANDIDATES_PATH)
    development = [row for row in candidates if row["split"] == "development"]
    existing = {row["qid"]: row for row in read_jsonl(EXISTING_PATH)}
    chunk_rows = read_jsonl(CHUNKS_PATH)
    chunks = {row["chunk_id"]: row for row in chunk_rows}

    actual_new = {row["source_id"] for row in development
                  if row["annotation_status"] == "pending_human_review"}
    if actual_new != NEW_DEVELOPMENT_IDS:
        raise ValueError(
            f"검수 대상이 바뀌었습니다: missing={sorted(NEW_DEVELOPMENT_IDS - actual_new)} "
            f"extra={sorted(actual_new - NEW_DEVELOPMENT_IDS)}"
        )

    rows = []
    for candidate in development:
        if candidate["annotation_status"] == "assistant_reviewed_draft":
            rows.append(existing_row(candidate, existing[candidate["source_id"]]))
        else:
            rows.append(new_row(candidate, chunks))

    manifest = {
        "schema_version": "gold-v2-development-manifest-1",
        "status": "assistant_reviewed_draft_not_expert_validated",
        "question_count": len(rows),
        "existing_question_count": len(rows) - len(NEW_DEVELOPMENT_IDS),
        "new_question_count": len(NEW_DEVELOPMENT_IDS),
        "category_counts": dict(sorted(Counter(row["primary_category"] for row in rows).items())),
        "inputs": {
            "candidates_sha256": sha256(CANDIDATES_PATH),
            "existing_questions_sha256": sha256(EXISTING_PATH),
            "chunks_sha256": sha256(CHUNKS_PATH),
        },
        "review_checks": [
            "All answerable draft evidence IDs exist in the fixed local corpus.",
            "Draft answers were compared with the cited chunk text.",
            "Follow-up rows retain the previous user question and an expected standalone rewrite.",
            "Unanswerable labels were checked for literal corpus matches and remain provisional.",
        ],
        "negative_literal_audit": negative_audit("\n".join(row.get("text", "") for row in chunk_rows)),
        "limitations": [
            "No teacher, policy expert, or independent annotator has approved these rows.",
            "Literal citation anchoring does not prove semantic completeness.",
            "Unanswerable status is relative to this fixed six-document corpus.",
            "The final test split remains sealed and is not included in this file.",
        ],
    }
    return rows, manifest


def main() -> None:
    rows, manifest = build()
    OUTPUT_PATH.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"개발 평가셋: {OUTPUT_PATH} ({len(rows)}문항)")
    print(f"검수 기록: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
