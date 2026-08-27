"""경로 부착(contextual retrieval)이 검색에 미치는 영향."""

import json
import pathlib

import numpy as np
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CHUNKS = pathlib.Path("data/processed/chunks.jsonl")
QUESTIONS = pathlib.Path("eval/questions.jsonl")
OUT = pathlib.Path("experiments/ablation_path.json")

MODEL = "BAAI/bge-m3"
KS = [1, 5, 10]
KEEP_POS = {"NNG", "NNP", "NNB", "NR", "SL", "SN", "SH", "VV", "VA", "MAG"}
kiwi = Kiwi()


def tokenize(t):
    return [x.form for x in kiwi.tokenize(t) if x.tag in KEEP_POS]


def evaluate(ranked_ids, gold):
    m = {}
    for k in KS:
        m[f"recall@{k}"] = len(gold & set(ranked_ids[:k])) / len(gold)
    m["mrr"] = next((1 / i for i, c in enumerate(ranked_ids, 1) if c in gold), 0.0)
    return m


def main():
    chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
    questions = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8")]
    questions = [q for q in questions if q["type"] != "unans"]
    ids = [c["chunk_id"] for c in chunks]

    variants = {
        "body": [c["body"] for c in chunks],
        "path": [c["path"] for c in chunks],
        "both": [c["path"] + "\n\n" + c["body"] for c in chunks],
    }

    model = SentenceTransformer(MODEL)
    qvecs = model.encode([q["question"] for q in questions],
                         normalize_embeddings=True)

    results = {}
    for name, texts in variants.items():
        print(f"\n[{name}] 색인 중...")
        bm25 = BM25Okapi([tokenize(t) for t in texts])

        cache = pathlib.Path(f"data/processed/emb_{name}.npy")
        if cache.exists():
            mat = np.load(cache)
            print(f"  임베딩 캐시 사용")
        else:
            print(f"  임베딩 생성 중...")
            mat = model.encode(texts, batch_size=8,
                               normalize_embeddings=True,
                               show_progress_bar=True)
            np.save(cache, mat)

        rows = []
        for q, qv in zip(questions, qvecs):
            gold = set(q["gold_chunks"])

            bs = bm25.get_scores(tokenize(q["question"]))
            bm_ids = [ids[i] for i in np.argsort(-bs)[:20]]

            ds = mat @ qv
            de_ids = [ids[i] for i in np.argsort(-ds)[:20]]

            rows.append({
                "qid": q["qid"], "type": q["type"],
                "bm25": evaluate(bm_ids, gold),
                "dense": evaluate(de_ids, gold),
            })
        results[name] = rows

    def avg(rows, method, key):
        return sum(r[method][key] for r in rows) / len(rows)

    for method in ["bm25", "dense"]:
        print(f"\n{'='*58}\n{method.upper()} — 경로 부착 효과\n{'='*58}")
        print(f"{'색인':8}{'R@1':>9}{'R@5':>9}{'R@10':>9}{'MRR':>9}")
        for name in variants:
            rows = results[name]
            print(f"{name:8}"
                  + "".join(f"{avg(rows, method, f'recall@{k}'):>9.3f}" for k in KS)
                  + f"{avg(rows, method, 'mrr'):>9.3f}")

    print(f"\n{'='*58}\n질문별 R@5 (Dense)\n{'='*58}")
    print(f"{'qid':8}{'type':7}" + "".join(f"{n:>8}" for n in variants))
    for i, q in enumerate(questions):
        row = f"{q['qid']:8}{q['type']:7}"
        for name in variants:
            row += f"{results[name][i]['dense']['recall@5']:>8.2f}"
        print(row)

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()