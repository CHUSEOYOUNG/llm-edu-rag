"""Build the 50-question V2 annotation seed without promoting drafts to gold."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXISTING_PATH = ROOT / "eval/questions.v2.draft.jsonl"
ANSWERABILITY_PATH = ROOT / "eval/answerability.v2.jsonl"
OUTPUT_DIR = ROOT / "eval/gold_v2"
OUTPUT_PATH = OUTPUT_DIR / "candidates.jsonl"
PLAN_PATH = OUTPUT_DIR / "plan.json"

CURRICULUM = "(2022 개정) 초·중등학교 교육과정 [별책1] 총론_ 국가교육위원회 고시 제2026-1호(2026.1.21.)"
HIGH = "2026 학교생활기록부 기재요령(고)_F_260219"


def chunk(doc: str, suffix: str) -> str:
    return f"{doc}::{suffix}"


PLAN = {
    "schema_version": "gold-v2-plan-1",
    "status": "annotation_seed_not_expert_validated",
    "target_count": 50,
    "split_policy": {
        "development": "May be used for debugging and model or threshold selection.",
        "test": "Sealed while pending review; must not be used for implementation decisions.",
    },
    "category_quotas": {
        "single_fact": {"development": 8, "test": 2},
        "multi_evidence": {"development": 8, "test": 2},
        "table_numeric": {"development": 6, "test": 4},
        "follow_up": {"development": 5, "test": 5},
        "unanswerable": {"development": 8, "test": 2},
    },
    "promotion_rule": (
        "A pending row becomes scoreable only after a human checks the question, "
        "reference answer, evidence, category, and split and changes annotation_status."
    ),
}


EXISTING_CATEGORIES = {
    "q001": "single_fact",
    "q002": "single_fact",
    "q003": "single_fact",
    "q004": "multi_evidence",
    "q005": "single_fact",
    "q006": "multi_evidence",
    "q007": "multi_evidence",
    "q008": "multi_evidence",
    "q009": "multi_evidence",
    "q010": "table_numeric",
    "q011": "table_numeric",
    "q012": "unanswerable",
    "q013": "unanswerable",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evidence_ids(row: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for group in row.get("evidence_groups", []):
        for alternative in group["alternatives"]:
            cid = alternative["chunk_id"]
            if cid not in ids:
                ids.append(cid)
    return ids or list(row.get("gold_chunks", []))


def candidate(
    *,
    source_id: str,
    split: str,
    category: str,
    question: str,
    answer: str | None,
    evidence: list[str],
    fact_key: str,
    origin: str,
    context: list[dict[str, str]] | None = None,
    standalone_question: str | None = None,
    annotation_status: str = "pending_human_review",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": "gold-v2-candidate-1",
        "candidate_id": "",  # assigned after deterministic ordering
        "source_id": source_id,
        "split": split,
        "release_status": "sealed" if split == "test" else "open",
        "primary_category": category,
        "expected_status": "unanswerable" if answer is None else "answerable",
        "question": question,
        "conversation_context": context or [],
        "standalone_question": standalone_question or question,
        "draft_reference_answer": answer,
        "candidate_evidence_chunk_ids": evidence,
        "fact_key": fact_key,
        "annotation_status": annotation_status,
        "evaluation_role": (
            "development_regression_only"
            if annotation_status == "assistant_reviewed_draft"
            else "withheld_final_test_candidate"
            if split == "test"
            else "development_candidate"
        ),
        "source_origin": origin,
        "review_notes": notes,
    }


def existing_candidates() -> list[dict[str, Any]]:
    rows = []
    for source in read_jsonl(EXISTING_PATH):
        qid = source["qid"]
        answer = None if qid in {"q012", "q013"} else source.get("proposed_answer", source["answer"])
        rows.append(candidate(
            source_id=qid,
            split="development",
            category=EXISTING_CATEGORIES[qid],
            question=source["question"],
            answer=answer,
            evidence=[] if answer is None else evidence_ids(source),
            fact_key=f"legacy_{qid}",
            origin="existing_reviewed_draft",
            annotation_status="assistant_reviewed_draft",
            notes="Existing V2 draft retained for regression; expert review is still pending.",
        ))
    return rows


def answerability_candidate(
    by_id: dict[str, dict[str, Any]], source_id: str, *, category: str,
    answer: str, fact_key: str, evidence_override: list[str] | None = None,
) -> dict[str, Any]:
    source = by_id[source_id]
    return candidate(
        source_id=source_id,
        split=source["split"],
        category=category,
        question=source["question"],
        answer=answer,
        evidence=evidence_override or source["evidence_chunk_ids"],
        fact_key=fact_key,
        origin="answerability_candidate",
        notes=(
            "Question came from answerability.v2; evidence was corrected to a chunk whose body contains the claim. "
            if evidence_override else "Question and evidence came from answerability.v2; "
        ) + "The draft answer still needs human review.",
    )


def new_candidates() -> list[dict[str, Any]]:
    answerability = {row["qid"]: row for row in read_jsonl(ANSWERABILITY_PATH)}
    rows = [
        answerability_candidate(answerability, "a006", category="single_fact",
            answer="결석을 시작한 날부터 5일 이내에 결석 신고서와 증빙서류를 제출해 학교장의 승인을 받아야 한다.",
            fact_key="illness_absence_submission_deadline"),
        answerability_candidate(answerability, "a007", category="single_fact",
            answer="학교장이 정한 등교 시각까지 출석하지 않은 경우를 지각으로 본다.",
            fact_key="attendance_late_definition",
            evidence_override=[chunk(HIGH, "s133::8조::0")]),
        answerability_candidate(answerability, "a008", category="single_fact",
            answer="입학 당시 주소 뒤에 변경된 주소를 누가하여 입력한다.",
            fact_key="student_record_address_change"),
        answerability_candidate(answerability, "a015", category="single_fact",
            answer="수업 시간의 일부 또는 전부에 불참한 경우를 결과로 본다.",
            fact_key="attendance_class_absence_definition",
            evidence_override=[chunk(HIGH, "s133::8조::0")]),
        answerability_candidate(answerability, "a016", category="single_fact",
            answer="형제나 자매의 결혼은 1일을 출석 인정 결석으로 처리한다.",
            fact_key="sibling_marriage_attendance_days"),
        candidate(
            source_id="authored-s001", split="development", category="single_fact",
            question="학년을 수료하려면 수업일수의 몇 분의 몇 이상 출석해야 하나요?",
            answer="각 학년 수료에는 수업일수의 3분의 2 이상 출석해야 한다.",
            evidence=[chunk(HIGH, "s133::8조::0")], fact_key="grade_completion_attendance_ratio",
            origin="corpus_authored_candidate"),

        candidate(
            source_id="authored-m001", split="development", category="multi_evidence",
            question="학생선수가 대회나 훈련으로 빠질 때 학교급별 허용일수와 지각·조퇴·결과의 일수 환산 기준은 어떻게 되나요?",
            answer="출석 인정 결석 허용 범위는 초등학교 20일, 중학교 35일, 고등학교 50일이며, 대회·훈련으로 인한 지각·조퇴·결과는 6시간 누적 시 1일로 본다.",
            evidence=[chunk(HIGH, "s136::나::1"), chunk(HIGH, "s136::나::2")],
            fact_key="student_athlete_days_and_hour_conversion", origin="corpus_authored_candidate"),
        candidate(
            source_id="authored-m002", split="development", category="multi_evidence",
            question="만성질환 학생이 질병결석으로 인정받으려면 무엇을 언제 제출해야 하며, 같은 학기에도 매번 진단서를 내야 하나요?",
            answer="의사의 진단서 또는 소견서를 첨부한 결석 신고서를 결석을 시작한 날부터 5일 이내에 제출해 학교장의 승인을 받아야 한다. 장기 치료가 필요한 경우 학기 초 최초 제출한 진단서로 해당 학기의 증빙을 갈음할 수 있다.",
            evidence=[chunk(HIGH, "s138::다::0"), chunk(HIGH, "s138::다::1")],
            fact_key="chronic_illness_absence_evidence", origin="corpus_authored_candidate"),
        candidate(
            source_id="authored-m003", split="development", category="multi_evidence",
            question="지난 학년도 생기부를 정정할 때 필요한 절차와 당시 입력 교사가 없을 경우 정정내용을 정하는 방법을 알려주세요.",
            answer="객관적인 증빙자료를 첨부하고 학업성적관리위원회 심의와 정정대장 결재 절차를 거쳐야 한다. 당시 입력 주체가 없어 초안 작성이 어려우면 필요한 자료를 토대로 학업성적관리위원회가 정정내용을 결정한다.",
            evidence=[chunk(HIGH, "s225::19조::0"), chunk(HIGH, "s225::19조::1")],
            fact_key="record_correction_process_and_absent_author", origin="corpus_authored_candidate"),
        candidate(
            source_id="authored-m004", split="test", category="multi_evidence",
            question="고등학교에서 1학점의 수업량과 졸업에 필요한 최소 이수 학점은 각각 얼마인가요?",
            answer="1학점은 50분 수업 16회에 해당하며, 고등학교 졸업을 위한 최소 이수 학점은 192학점이다.",
            evidence=[chunk(CURRICULUM, "s026::2::0"), chunk(CURRICULUM, "s026::2::1")],
            fact_key="high_school_credit_workload_and_graduation_total", origin="corpus_authored_candidate"),
        candidate(
            source_id="authored-m005", split="test", category="multi_evidence",
            question="고등학교 창의적 체험활동의 세 영역과 봉사활동·진로활동에 기록하는 내용을 함께 알려주세요.",
            answer="창의적 체험활동은 자율·자치활동, 동아리활동, 진로활동으로 구성된다. 봉사활동에는 일자 또는 기간, 장소나 주관기관, 활동내용과 시간을 기록하고, 진로활동 특기사항에는 진로희망분야, 진로검사·상담 결과와 관련 활동 등 진로 특성이 드러나는 내용을 기록한다.",
            evidence=[chunk(HIGH, "s154::13조::0"), chunk(HIGH, "s154::13조::1")],
            fact_key="high_school_creative_activity_record_fields", origin="corpus_authored_candidate"),

        candidate(source_id="authored-t001", split="development", category="table_numeric",
            question="초등학교 3~4학년군 수학의 기준 수업 시수는 몇 시간인가요?",
            answer="2년간 272시간이다.", evidence=[chunk(CURRICULUM, "s020::2::0")],
            fact_key="elementary_34_math_hours", origin="corpus_authored_candidate"),
        candidate(source_id="authored-t002", split="development", category="table_numeric",
            question="초등학교 5~6학년군 영어의 기준 수업 시수는 몇 시간인가요?",
            answer="2년간 204시간이다.", evidence=[chunk(CURRICULUM, "s020::2::0")],
            fact_key="elementary_56_english_hours", origin="corpus_authored_candidate"),
        candidate(source_id="authored-t003", split="development", category="table_numeric",
            question="중학교 수학의 3년간 기준 수업 시수는 몇 시간인가요?",
            answer="3년간 374시간이다.", evidence=[chunk(CURRICULUM, "s023::2::0")],
            fact_key="middle_math_hours", origin="corpus_authored_candidate"),
        candidate(source_id="authored-t004", split="development", category="table_numeric",
            question="중학교 3년간 최소 총 수업 시간은 몇 시간인가요?",
            answer="3년간 최소 3,366시간이다.", evidence=[chunk(CURRICULUM, "s023::2::0")],
            fact_key="middle_total_minimum_hours", origin="corpus_authored_candidate"),
        candidate(source_id="authored-t005", split="test", category="table_numeric",
            question="일반고에서 과학 교과군의 필수 이수 학점은 몇 학점인가요?",
            answer="10학점이다.", evidence=[chunk(CURRICULUM, "s026::2::0")],
            fact_key="general_high_science_required_credits", origin="corpus_authored_candidate"),
        candidate(source_id="authored-t006", split="test", category="table_numeric",
            question="일반고 창의적 체험활동의 최소 이수 학점과 환산 시간은 얼마인가요?",
            answer="18학점이며 288시간이다.", evidence=[chunk(CURRICULUM, "s026::2::0")],
            fact_key="general_high_creative_activity_credits_hours", origin="corpus_authored_candidate"),
        candidate(source_id="authored-t007", split="test", category="table_numeric",
            question="특수목적고와 자율고는 예술 교과군을 최소 몇 학점 이수해야 하나요?",
            answer="5학점 이상 이수해야 한다.", evidence=[chunk(CURRICULUM, "s026::2::1")],
            fact_key="special_autonomous_high_arts_minimum", origin="corpus_authored_candidate"),
        candidate(source_id="authored-t008", split="test", category="table_numeric",
            question="산업수요 맞춤형고와 특성화고에서 전문 교과군의 필수 이수 학점은 몇 학점인가요?",
            answer="80학점이다.", evidence=[chunk(CURRICULUM, "s027::나::0")],
            fact_key="vocational_high_professional_subject_credits", origin="corpus_authored_candidate"),

        follow_up("authored-f001", "development",
            "초등학교 3~4학년군 국어 수업 시수는 몇 시간이야?", "초등학교 3~4학년군 국어는 408시간이야.", "그럼 5~6학년은?",
            "초등학교 5~6학년군 국어의 기준 수업 시수는 몇 시간인가요?", "2년간 408시간이다.",
            [chunk(CURRICULUM, "s020::2::0")], "followup_elementary_56_korean_hours"),
        follow_up("authored-f002", "development",
            "초등학교 수업 한 시간은 몇 분이야?", "초등학교 수업 한 시간은 40분이 원칙이야.", "중학교는?",
            "중학교 수업 한 시간은 몇 분이 원칙인가요?", "45분이 원칙이다.",
            [chunk(CURRICULUM, "s023::2::0")], "followup_middle_class_minutes"),
        follow_up("authored-f003", "development",
            "지난 학년도 생기부도 정정할 수 있어?", "지난 생기부도 객관적인 증빙자료가 있으면 정정 절차를 밟을 수 있어.", "증빙자료가 없으면?",
            "지난 학년도 학교생활기록부는 객관적인 증빙자료가 없어도 정정할 수 있나요?",
            "원칙적으로 정정할 수 없다.", [chunk(HIGH, "s225::19조::0")],
            "followup_record_correction_without_evidence"),
        follow_up("authored-f004", "development",
            "중학교 선택 교과에는 어떤 과목이 있어?", "중학교 선택 교과에는 한문, 환경, 생활 외국어, 보건, 진로와 직업 등이 있어.", "생활 외국어에는 뭐가 있어?",
            "중학교 선택 교과의 생활 외국어에는 어떤 과목이 있나요?",
            "생활 독일어, 프랑스어, 스페인어, 중국어, 일본어, 러시아어, 아랍어, 베트남어가 있다.",
            [chunk(CURRICULUM, "s022::1::0")], "followup_middle_living_foreign_languages"),
        follow_up("authored-f005", "development",
            "초등학교 정보교육은 최소 몇 시간이야?", "초등학교 정보교육은 34시간 이상 편성해.", "그럼 중학교는?",
            "중학교 정보교육은 최소 몇 시간을 편성해야 하나요?", "68시간 이상 편성해야 한다.",
            [chunk(CURRICULUM, "s023::2::0")], "followup_middle_information_hours"),
        follow_up("authored-f006", "test",
            "고등학교 수업 한 시간은 몇 분이야?", "고등학교 수업 한 시간은 50분이 원칙이야.", "그럼 1학점은 몇 번 수업해?",
            "고등학교에서 1학점은 50분 수업 몇 회에 해당하나요?", "16회에 해당한다.",
            [chunk(CURRICULUM, "s026::2::0")], "followup_high_credit_sessions"),
        follow_up("authored-f007", "test",
            "독서활동상황에는 무엇을 입력해?", "독서활동상황에는 학생이 읽은 책의 제목과 저자를 입력해.", "누가 입력해?",
            "중·고등학교 독서활동상황의 책 제목과 저자는 누가 입력하나요?",
            "교과 담당교사 또는 학급 담임교사가 입력한다.", [chunk(HIGH, "s086::15조의3::0")],
            "followup_reading_activity_author"),
        follow_up("authored-f008", "test",
            "중학교 학교스포츠클럽은 언제 편성해?", "중학교 학교스포츠클럽 활동은 매 학기 편성해.", "한 해에 몇 시간이야?",
            "중학교 학교스포츠클럽 활동은 학년별로 연간 몇 시간 운영하나요?",
            "학년별 연간 34~68시간 운영한다.", [chunk(CURRICULUM, "s023::2::3")],
            "followup_middle_sports_club_annual_hours"),
        follow_up("authored-f009", "test",
            "일반고 과학은 몇 학점 이수해야 해?", "일반고 과학 교과군의 필수 이수 학점은 10학점이야.", "한국사는?",
            "일반고에서 한국사의 필수 이수 학점은 몇 학점인가요?", "6학점이다.",
            [chunk(CURRICULUM, "s026::2::0")], "followup_general_high_korean_history_credits"),
        follow_up("authored-f010", "test",
            "특수목적고와 자율고는 예술을 몇 학점 이수해야 해?", "특수목적고와 자율고는 예술 교과군을 5학점 이상 이수해야 해.", "기술·가정이나 정보 쪽은?",
            "특수목적고와 자율고에서 기술·가정, 정보, 제2외국어, 한문, 교양 교과군은 최소 몇 학점 이수해야 하나요?",
            "12학점 이상 이수해야 한다.", [chunk(CURRICULUM, "s026::2::1")],
            "followup_special_autonomous_high_other_subject_credits"),
    ]

    unanswerable_ids = {
        "development": ["u002", "u003", "u004", "u010", "u011", "u012"],
        "test": ["u014", "u016"],
    }
    for split, source_ids in unanswerable_ids.items():
        for source_id in source_ids:
            source = answerability[source_id]
            rows.append(candidate(
                source_id=source_id,
                split=split,
                category="unanswerable",
                question=source["question"],
                answer=None,
                evidence=[],
                fact_key=f"unanswerable_{source_id}",
                origin="answerability_candidate",
                notes="Expected to be absent from the fixed corpus; a human must confirm corpus coverage.",
            ))
    return rows


def follow_up(
    source_id: str, split: str, previous_question: str, previous_answer: str, current_question: str,
    standalone_question: str, answer: str, evidence: list[str], fact_key: str,
) -> dict[str, Any]:
    return candidate(
        source_id=source_id,
        split=split,
        category="follow_up",
        question=current_question,
        context=[
            {"role": "user", "content": previous_question},
            {"role": "assistant", "content": previous_answer},
        ],
        standalone_question=standalone_question,
        answer=answer,
        evidence=evidence,
        fact_key=fact_key,
        origin="corpus_authored_candidate",
        notes="Evaluate both context resolution and retrieval; do not score as a standalone short query.",
    )


def build() -> list[dict[str, Any]]:
    rows = existing_candidates() + new_candidates()
    for index, row in enumerate(rows, start=1):
        row["candidate_id"] = f"v2q{index:03d}"
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build()
    PLAN_PATH.write_text(json.dumps(PLAN, ensure_ascii=False, indent=2) + "\n")
    OUTPUT_PATH.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"작성: {OUTPUT_PATH} ({len(rows)}문항)")
    print(f"설계: {PLAN_PATH}")


if __name__ == "__main__":
    main()
