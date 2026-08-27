"""Dense 검색 (BGE-m3) + BM25와 동일 지표 측정."""

import json
import pathlib

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS = pathlib.Path("data/processed/chunks.jsonl")
QUESTIONS = pathlib.Path("eval/questions.jsonl")
EMB = pathlib.Path("data/processed/embeddings.npy")
OUT = pathlib.Path("experiments/dense.json")
OUT.parent.mkdir(exist_ok=True)

MODEL = "BAAI/bge-m3"
KS = [1, 3, 5, 10]


def evaluate(ranked_ids, gold):
    res = {}
    for k in KS:
        topk = ranked_ids[:k]
        res[f"hit@{k}"] = 1.0 if any(c in gold for c in topk) else 0.0
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
    ids = [c["chunk_id"] for c in chunks]
    by_id = {c["chunk_id"]: c for c in chunks}

    print(f"모델 로드: {MODEL}")
    model = SentenceTransformer(MODEL)

    if EMB.exists():
        print(f"기존 임베딩 사용: {EMB}")
        mat = np.load(EMB)
        if len(mat) != len(chunks):
            print("  청크 수 불일치 → 재생성")
            mat = None
    else:
        mat = None

    if mat is None:
        print(f"임베딩 생성 중 ({len(chunks)}개)... 5~10분 소요")
        texts = [c["text"] for c in chunks]     # 경로 포함 본문
        mat = model.encode(
            texts, batch_size=8, normalize_embeddings=True,
            show_progress_bar=True,
        )
        np.save(EMB, mat)
        print(f"저장: {EMB}")

    targets = [q for q in questions if q["type"] != "unans"]
    qvecs = model.encode(
        [q["question"] for q in targets], normalize_embeddings=True
    )

    from collections import defaultdict
    per_type = defaultdict(list)
    rows = []

    for q, qv in zip(targets, qvecs):
        gold = set(q["gold_chunks"])
        sims = mat @ qv                          # 정규화했으므로 내적 = 코사인
        order = np.argsort(-sims)[:20]
        ranked_ids = [ids[i] for i in order]

        m = evaluate(ranked_ids, gold)
        per_type[q["type"]].append(m)
        rows.append({"qid": q["qid"], "type": q["type"], **m,
                     "top1": ranked_ids[0], "top1_score": float(sims[order[0]])})

    def avg(ms, key):
        return sum(m[key] for m in ms) / len(ms)

    allm = [m for ms in per_type.values() for m in ms]

    print(f"\n{'='*54}\nDense / {MODEL} (질문 {len(allm)}개)\n{'='*54}")
    print(f"{'':8}" + "".join(f"{'R@'+str(k):>8}" for k in KS) + f"{'MRR':>8}")
    print(f"{'전체':8}" + "".join(f"{avg(allm, f'recall@{k}'):>8.3f}" for k in KS)
          + f"{avg(allm, 'mrr'):>8.3f}")
    for t in ["fact", "cond", "table"]:
        if t in per_type:
            ms = per_type[t]
            print(f"{t:8}" + "".join(f"{avg(ms, f'recall@{k}'):>8.3f}" for k in KS)
                  + f"{avg(ms, 'mrr'):>8.3f}  (n={len(ms)})")
    print(f"\nHit@5: {avg(allm, 'hit@5'):.3f}")

    # BM25와 비교
    bm = pathlib.Path("experiments/bm25.json")
    if bm.exists():
        b = json.loads(bm.read_text(encoding="utf-8"))
        bq = {r["qid"]: r for r in b["per_question"]}
        print(f"\n--- BM25 대비 (R@5) ---")
        for r in rows:
            before = bq[r["qid"]]["recall@5"]
            after = r["recall@5"]
            d = after - before
            mark = "↑" if d > 0 else ("↓" if d < 0 else " ")
            print(f"{mark} {r['qid']} ({r['type']:5}) {before:.2f} → {after:.2f}")
        print(f"\n전체 R@5: {b['overall']['recall@5']:.3f} → {avg(allm, 'recall@5'):.3f}")

    print("\n--- 실패 ---")
    for r in rows:
        if r["recall@5"] == 0:
            print(f"X {r['qid']} ({r['type']})")
            print(f"    top1({r['top1_score']:.3f}): {by_id[r['top1']]['path'][:60]}")

    OUT.write_text(json.dumps({
        "method": "dense",
        "model": MODEL,
        "n_questions": len(allm),
        "overall": {k: avg(allm, k) for k in allm[0]},
        "per_type": {t: {k: avg(ms, k) for k in ms[0]} for t, ms in per_type.items()},
        "per_question": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main() 