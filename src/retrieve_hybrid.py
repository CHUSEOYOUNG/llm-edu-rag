"""Hybrid 검색 (RRF) — BM25 + Dense 순위 결합."""

import json
import pathlib
from collections import defaultdict

import numpy as np
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CHUNKS = pathlib.Path("data/processed/chunks.jsonl")
QUESTIONS = pathlib.Path("eval/questions.jsonl")
EMB = pathlib.Path("data/processed/embeddings.npy")
OUT = pathlib.Path("experiments/hybrid.json")

MODEL = "BAAI/bge-m3"
KS = [1, 3, 5, 10]
RRF_K = 60          # RRF 상수. 클수록 하위 순위 영향 ↑
TOP_N = 50          # 각 방식에서 가져올 후보 수
KEEP_POS = {"NNG", "NNP", "NNB", "NR", "SL", "SN", "SH", "VV", "VA", "MAG"}

kiwi = Kiwi()


def tokenize(text):
    return [t.form for t in kiwi.tokenize(text) if t.tag in KEEP_POS]


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


def rrf(rank_lists, k=RRF_K):
    """Reciprocal Rank Fusion: score = sum(1 / (k + rank))"""
    scores = defaultdict(float)
    for lst in rank_lists:
        for rank, cid in enumerate(lst, 1):
            scores[cid] += 1 / (k + rank)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])]


def main():
    chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
    questions = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8")]
    ids = [c["chunk_id"] for c in chunks]

    print("BM25 색인...")
    bm25 = BM25Okapi([tokenize(c["body"]) for c in chunks])

    print("임베딩 로드...")
    mat = np.load(EMB)
    model = SentenceTransformer(MODEL)

    targets = [q for q in questions if q["type"] != "unans"]
    qvecs = model.encode([q["question"] for q in targets],
                         normalize_embeddings=True)

    per_type = defaultdict(list)
    rows = []

    for q, qv in zip(targets, qvecs):
        gold = set(q["gold_chunks"])

        bs = bm25.get_scores(tokenize(q["question"]))
        bm_rank = [ids[i] for i in np.argsort(-bs)[:TOP_N]]

        ds = mat @ qv
        de_rank = [ids[i] for i in np.argsort(-ds)[:TOP_N]]

        fused = rrf([bm_rank, de_rank])[:20]
        m = evaluate(fused, gold)
        per_type[q["type"]].append(m)
        rows.append({"qid": q["qid"], "type": q["type"], **m})

    def avg(ms, key):
        return sum(m[key] for m in ms) / len(ms)

    allm = [m for ms in per_type.values() for m in ms]

    print(f"\n{'='*54}\nHybrid RRF (k={RRF_K}, 질문 {len(allm)}개)\n{'='*54}")
    print(f"{'':8}" + "".join(f"{'R@'+str(k):>8}" for k in KS) + f"{'MRR':>8}")
    print(f"{'전체':8}" + "".join(f"{avg(allm, f'recall@{k}'):>8.3f}" for k in KS)
          + f"{avg(allm, 'mrr'):>8.3f}")
    for t in ["fact", "cond", "table"]:
        if t in per_type:
            ms = per_type[t]
            print(f"{t:8}" + "".join(f"{avg(ms, f'recall@{k}'):>8.3f}" for k in KS)
                  + f"{avg(ms, 'mrr'):>8.3f}  (n={len(ms)})")

    # 3방식 비교
    print(f"\n{'='*54}\n3방식 비교 (R@5)\n{'='*54}")
    prev = {}
    for name in ["bm25", "dense"]:
        p = pathlib.Path(f"experiments/{name}.json")
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            prev[name] = {r["qid"]: r["recall@5"] for r in d["per_question"]}
            prev[name]["_all"] = d["overall"]["recall@5"]

    print(f"{'qid':6}{'type':7}{'BM25':>8}{'Dense':>8}{'Hybrid':>8}")
    for r in rows:
        b = prev.get("bm25", {}).get(r["qid"], float("nan"))
        d = prev.get("dense", {}).get(r["qid"], float("nan"))
        print(f"{r['qid']:6}{r['type']:7}{b:>8.2f}{d:>8.2f}{r['recall@5']:>8.2f}")
    print(f"{'전체':6}{'':7}{prev.get('bm25',{}).get('_all',0):>8.3f}"
          f"{prev.get('dense',{}).get('_all',0):>8.3f}"
          f"{avg(allm, 'recall@5'):>8.3f}")

    OUT.write_text(json.dumps({
        "method": "hybrid_rrf",
        "rrf_k": RRF_K,
        "index_text": "body",
        "candidate_depth": TOP_N,
        "evaluation_depth": 20,
        "n_questions": len(allm),
        "overall": {k: avg(allm, k) for k in allm[0]},
        "per_type": {t: {k: avg(ms, k) for k in ms[0]} for t, ms in per_type.items()},
        "per_question": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()