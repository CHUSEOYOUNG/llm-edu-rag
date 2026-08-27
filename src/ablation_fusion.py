"""융합 방식 비교: RRF vs Weighted Sum (가중치 스윕)."""

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
OUT = pathlib.Path("experiments/ablation_fusion.json")

MODEL = "BAAI/bge-m3"
KS = [1, 5, 10]
KEEP_POS = {"NNG", "NNP", "NNB", "NR", "SL", "SN", "SH", "VV", "VA", "MAG"}
kiwi = Kiwi()


def tokenize(t):
    return [x.form for x in kiwi.tokenize(t) if x.tag in KEEP_POS]


def evaluate(ranked, gold):
    m = {}
    for k in KS:
        m[f"recall@{k}"] = len(gold & set(ranked[:k])) / len(gold)
    m["mrr"] = next((1 / i for i, c in enumerate(ranked, 1) if c in gold), 0.0)
    return m


def minmax(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def main():
    chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
    questions = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8")]
    questions = [q for q in questions if q["type"] != "unans"]
    ids = [c["chunk_id"] for c in chunks]

    print("색인 중...")
    bm25 = BM25Okapi([tokenize(c["body"]) for c in chunks])
    mat = np.load(EMB)
    model = SentenceTransformer(MODEL)
    qvecs = model.encode([q["question"] for q in questions],
                         normalize_embeddings=True)

    # 질의별 점수 미리 계산
    scored = []
    for q, qv in zip(questions, qvecs):
        scored.append({
            "q": q,
            "bm": bm25.get_scores(tokenize(q["question"])),
            "de": mat @ qv,
        })

    methods = {}

    # RRF (k=60)
    def run_rrf(k=60):
        rows = []
        for s in scored:
            r = defaultdict(float)
            for arr in (s["bm"], s["de"]):
                for rank, i in enumerate(np.argsort(-arr)[:50], 1):
                    r[ids[i]] += 1 / (k + rank)
            ranked = [c for c, _ in sorted(r.items(), key=lambda x: -x[1])][:20]
            rows.append({"qid": s["q"]["qid"], "type": s["q"]["type"],
                         **evaluate(ranked, set(s["q"]["gold_chunks"]))})
        return rows

    methods["rrf"] = run_rrf()

    # Weighted sum: alpha * dense + (1-alpha) * bm25 (min-max 정규화 후)
    for alpha in [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]:
        rows = []
        for s in scored:
            fused = alpha * minmax(s["de"]) + (1 - alpha) * minmax(s["bm"])
            ranked = [ids[i] for i in np.argsort(-fused)[:20]]
            rows.append({"qid": s["q"]["qid"], "type": s["q"]["type"],
                         **evaluate(ranked, set(s["q"]["gold_chunks"]))})
        methods[f"w{alpha:.1f}"] = rows

    def avg(rows, key):
        return sum(r[key] for r in rows) / len(rows)

    print(f"\n{'='*58}\n융합 방식 비교 (질문 {len(questions)}개)\n{'='*58}")
    print(f"{'방식':10}{'R@1':>9}{'R@5':>9}{'R@10':>9}{'MRR':>9}")
    for name, rows in methods.items():
        label = name if name == "rrf" else f"α={name[1:]}"
        print(f"{label:10}"
              + "".join(f"{avg(rows, f'recall@{k}'):>9.3f}" for k in KS)
              + f"{avg(rows, 'mrr'):>9.3f}")

    print("\nα=0.0 은 BM25 단독, α=1.0 은 Dense 단독")

    OUT.write_text(json.dumps(methods, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()