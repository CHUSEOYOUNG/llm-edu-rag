"""평가셋의 gold_chunks를 대화형으로 채운다.

사용: uv run python src/find_gold.py
"""
import json
import pathlib
import re
import sys

CHUNKS = pathlib.Path("data/processed/chunks.jsonl")
QUESTIONS = pathlib.Path("eval/questions.jsonl")
TOP_N = 8


# ---------- 한국어 문자 bigram 기반 간이 검색 ----------
# 형태소 분석기 없이 동작. 나중에 붙일 BM25의 baseline 역할도 겸함.

_CLEAN = re.compile(r"[^가-힣a-zA-Z0-9]+")


def bigrams(text: str) -> set:
    t = _CLEAN.sub(" ", text)
    grams = set()
    for token in t.split():
        if len(token) == 1:
            grams.add(token)
        for i in range(len(token) - 1):
            grams.add(token[i:i + 2])
    return grams


def score(query_grams: set, doc_grams: set) -> float:
    if not query_grams:
        return 0.0
    return len(query_grams & doc_grams) / len(query_grams)


def search(query: str, chunks: list, indexed: list, n: int = TOP_N) -> list:
    qg = bigrams(query)
    scored = [(score(qg, dg), i) for i, dg in enumerate(indexed)]
    scored.sort(reverse=True)
    return [(s, chunks[i]) for s, i in scored[:n] if s > 0]


# ---------- 화면 출력 ----------

def show_candidates(results: list, selected: set):
    if not results:
        print("  (결과 없음)")
        return
    for i, (s, c) in enumerate(results):
        mark = "*" if c["chunk_id"] in selected else " "
        print(f"\n{mark}[{i}] {s:.2f} | {c['chunk_id']}")
        print(f"     경로: {c['path'][:70]}")
        body = c["body"][:180].replace("\n", " ")
        print(f"     {body}")


def show_question(q: dict, idx: int, total: int):
    print("\n" + "=" * 72)
    print(f"[{idx}/{total}] {q['qid']} ({q['type']})")
    print(f"질문: {q['question']}")
    if q.get("answer"):
        print(f"정답: {q['answer']}")
    hint = (q.get("meta") or {}).get("source_hint")
    if hint:
        print(f"힌트: {hint}")
    if q.get("gold_chunks"):
        print(f"현재 선택: {len(q['gold_chunks'])}개")


HELP = """
명령:
  0 3 5     번호 선택/해제 (토글, 공백 구분)
  s 키워드   해당 키워드로 재검색
  a         선택 확정하고 다음 질문
  n         건너뛰기
  d         현재 선택 전체 해제
  q         저장하고 종료
  ?         도움말
"""


def main():
    if not QUESTIONS.exists():
        sys.exit(f"{QUESTIONS} 없음. 평가셋 파일을 먼저 만드세요.")

    chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
    indexed = [bigrams(c["path"] + " " + c["body"]) for c in chunks]
    questions = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8")]

    print(f"청크 {len(chunks)}개 / 질문 {len(questions)}개 로드")
    print(HELP)

    targets = [q for q in questions
               if q["type"] != "unans" and not q.get("gold_chunks")]
    print(f"작업 대상: {len(targets)}개 (unans·완료분 제외)\n")

    for idx, q in enumerate(targets, 1):
        selected = set(q.get("gold_chunks") or [])
        query = q["question"] + " " + (q.get("answer") or "")
        results = search(query, chunks, indexed)

        show_question(q, idx, len(targets))
        show_candidates(results, selected)

        while True:
            cmd = input("\n> ").strip()

            if not cmd:
                continue
            if cmd == "?":
                print(HELP)
            elif cmd == "q":
                q["gold_chunks"] = sorted(selected)
                save(questions)
                print("저장 후 종료")
                return
            elif cmd == "n":
                break
            elif cmd == "a":
                if not selected:
                    print("  선택된 게 없습니다. 그래도 넘어가려면 'n'")
                    continue
                q["gold_chunks"] = sorted(selected)
                print(f"  → {len(selected)}개 저장")
                save(questions)
                break
            elif cmd == "d":
                selected.clear()
                show_candidates(results, selected)
            elif cmd.startswith("s "):
                results = search(cmd[2:], chunks, indexed)
                show_candidates(results, selected)
            else:
                            try:
                                for tok in cmd.split():
                                    if tok.startswith("-"):
                                        selected.discard(results[int(tok[1:])][1]["chunk_id"])
                                    else:
                                        selected.add(results[int(tok)][1]["chunk_id"])
                            except (ValueError, IndexError):
                                print("  잘못된 번호. '?' 로 도움말")
                                continue
                            show_candidates(results, selected)
                            print(f"  선택 {len(selected)}개")

    save(questions)
    print("\n완료.")


def save(questions: list):
    with QUESTIONS.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()