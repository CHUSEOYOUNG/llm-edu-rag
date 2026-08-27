"""검증된 gold_chunks를 questions.jsonl에 지정한다."""

import json
import pathlib

QUESTIONS = pathlib.Path("eval/questions.jsonl")
CHUNKS = pathlib.Path("data/processed/chunks.jsonl")

CR = "(2022 개정) 초·중등학교 교육과정 [별책1] 총론_ 국가교육위원회 고시 제2026-1호(2026.1.21.)"
CHO = "2026 학교생활기록부 기재요령(초)_F_260219"
JUNG = "2026 학교생활기록부 기재요령(중)_F_260227"
GO = "2026 학교생활기록부 기재요령(고)_F_260219"

GOLD = {
    "q001": [
        f"{CHO}::s433::22조::0",
        f"{JUNG}::s267::22조::0",
        f"{GO}::s061::22조::0",
    ],
    "q002": [
        f"{CHO}::s463::19조::0",
        f"{JUNG}::s298::19조::0",
        f"{GO}::s092::19조::0",
    ],
    "q003": [
        f"{JUNG}::s416::7::7",
        f"{GO}::s250::7::5",
    ],
    "q004": [f"{CR}::s020::2::1"],
    "q005": [f"{CR}::s022::1::0"],
    "q006": [f"{CR}::s020::2::0", f"{CR}::s023::2::0"],
    "q007": [f"{CR}::s020::2::0", f"{CR}::s023::2::0"],
    "q008": [f"{CR}::s019::1::0"],
    "q009": [f"{CR}::s019::1::0", f"{CR}::s022::1::0"],
    "q010": [f"{CR}::s023::2::0"],
    "q011": [f"{CR}::s020::2::0"],
}

NOTE_UPDATE = {
    "q003": "초등판은 동일 문구가 이미지 텍스트로 삽입되어 검색 불가. 문서군별 결손 사례",
    "q011": "표 파싱 붕괴로 셀 정렬 무너짐. q010(중학교, 정상)과 대조",
}


def main():
    chunk_ids = {json.loads(l)["chunk_id"] for l in CHUNKS.open(encoding="utf-8")}
    questions = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8")]
    by_qid = {q["qid"]: q for q in questions}

    missing_total = 0
    for qid, ids in GOLD.items():
        q = by_qid.get(qid)
        if not q:
            print(f"[{qid}] 질문 없음 — 건너뜀")
            continue

        missing = [c for c in ids if c not in chunk_ids]
        if missing:
            missing_total += len(missing)
            print(f"[{qid}] 존재하지 않는 chunk_id {len(missing)}개")
            for m in missing:
                print(f"        {m}")

        valid = [c for c in ids if c in chunk_ids]
        q["gold_chunks"] = sorted(valid)
        if qid in NOTE_UPDATE:
            q["note"] = NOTE_UPDATE[qid]
        print(f"[{qid}] gold {len(valid)}개")

    with QUESTIONS.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print()
    print(f"경고: 매칭 실패 {missing_total}건" if missing_total else "전체 매칭 성공")


if __name__ == "__main__":
    main()