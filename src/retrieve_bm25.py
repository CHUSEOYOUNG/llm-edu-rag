"""BM25 검색 + Recall@k / MRR / Hit Rate 측정."""

import json
import pathlib
from collections import defaultdict

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

CHUNKS = pathlib.Path("data/processed/chunks.jsonl")
QUESTIONS = pathlib.Path("eval/questions.jsonl")
OUT = pathlib.Path("experiments/bm25.json")
OUT.parent.mkdir(exist_ok=True)

KS = [1, 3, 5, 10]
# 조사·어미·기호는 검색에 방해가 되므로 제외
KEEP_POS = {"NNG", "NNP", "NNB", "NR", "SL", "SN", "SH", "VV", "VA", "MAG"}

kiwi = Kiwi()


def tokenize(text: str) -> list:
    return [t.form for t in kiwi.tokenize(text) if t.tag in KEEP_POS]


def evaluate(ranked_ids: list, gold: set) -> dict:
    """단일 질의의 지표 계산."""
    res = {}
    for k in KS:
        topk = ranked_ids[:k]
        hit = any(c in gold for c in topk)
        res[f"hit@{k}"] = 1.0 if hit else 0.0
        res[f"recall@{k}"] = len(gold & set(topk)) / len(gold)

    rr = 0.0
    for i, cid in enumerate(ranked_ids, 1):
        if cid in gold:
            rr = 1 / i
            break
    res["mrr"] = rr
    return res


def main():
    chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
    questions = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8")]

    print(f"청크 {len(chunks)}개 색인 중...")
    corpus = [tokenize(c["path"] + " " + c["body"]) for c in chunks]
    bm25 = BM25Okapi(corpus)
    ids = [c["chunk_id"] for c in chunks]
    by_id = {c["chunk_id"]: c for c in chunks}

    targets = [q for q in questions if q["type"] != "unans"]
    per_type = defaultdict(list)
    rows = []

    for q in targets:
        gold = set(q["gold_chunks"])
        scores = bm25.get_scores(tokenize(q["question"]))
        ranked = sorted(range(len(ids)), key=lambda i: -scores[i])
        ranked_ids = [ids[i] for i in ranked[:20]]

        m = evaluate(ranked_ids, gold)
        per_type[q["type"]].append(m)
        rows.append({"qid": q["qid"], "type": q["type"], **m,
                     "top1": ranked_ids[0]})

    def avg(ms, key):
        return sum(m[key] for m in ms) / len(ms)

    allm = [m for ms in per_type.values() for m in ms]

    print(f"\n{'='*54}\nBM25 (질문 {len(allm)}개)\n{'='*54}")
    print(f"{'':8}" + "".join(f"{'R@'+str(k):>8}" for k in KS) + f"{'MRR':>8}")
    print(f"{'전체':8}" + "".join(f"{avg(allm, f'recall@{k}'):>8.3f}" for k in KS)
          + f"{avg(allm, 'mrr'):>8.3f}")
    for t in ["fact", "cond", "table"]:
        if t in per_type:
            ms = per_type[t]
            print(f"{t:8}" + "".join(f"{avg(ms, f'recall@{k}'):>8.3f}" for k in KS)
                  + f"{avg(ms, 'mrr'):>8.3f}  (n={len(ms)})")

    print(f"\nHit@5: {avg(allm, 'hit@5'):.3f}")

    print("\n--- 질문별 ---")
    for r in sorted(rows, key=lambda x: x["recall@5"]):
        mark = "O" if r["recall@5"] > 0 else "X"
        print(f"{mark} {r['qid']} ({r['type']:5}) R@5={r['recall@5']:.2f} MRR={r['mrr']:.2f}")
        if r["recall@5"] == 0:
            print(f"    top1: {by_id[r['top1']]['path'][:60]}")

    OUT.write_text(json.dumps({
        "method": "bm25",
        "n_questions": len(allm),
        "overall": {k: avg(allm, k) for k in allm[0]},
        "per_type": {t: {k: avg(ms, k) for k in ms[0]} for t, ms in per_type.items()},
        "per_question": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()