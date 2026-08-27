"""실패 질의 진단: gold 청크가 실제로 몇 위에 있는지 확인."""

import json
import pathlib

import numpy as np
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CHUNKS = pathlib.Path("data/processed/chunks.jsonl")
QUESTIONS = pathlib.Path("eval/questions.jsonl")
EMB = pathlib.Path("data/processed/embeddings.npy")

TARGETS = ["q001", "q002", "q009"]
KEEP_POS = {"NNG", "NNP", "NNB", "NR", "SL", "SN", "SH", "VV", "VA", "MAG"}

kiwi = Kiwi()


def tokenize(t):
    return [x.form for x in kiwi.tokenize(t) if x.tag in KEEP_POS]


chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
questions = {json.loads(l)["qid"]: json.loads(l)
             for l in QUESTIONS.open(encoding="utf-8")}
ids = [c["chunk_id"] for c in chunks]
by_id = {c["chunk_id"]: c for c in chunks}
pos = {cid: i for i, cid in enumerate(ids)}

print("색인 중...")
bm25 = BM25Okapi([tokenize(c["path"] + " " + c["body"]) for c in chunks])
mat = np.load(EMB)
model = SentenceTransformer("BAAI/bge-m3")

for qid in TARGETS:
    q = questions[qid]
    gold = q["gold_chunks"]

    bs = bm25.get_scores(tokenize(q["question"]))
    bm_rank = {cid: r for r, cid in enumerate(
        [ids[i] for i in np.argsort(-bs)], 1)}

    qv = model.encode([q["question"]], normalize_embeddings=True)[0]
    ds = mat @ qv
    de_rank = {cid: r for r, cid in enumerate(
        [ids[i] for i in np.argsort(-ds)], 1)}

    print(f"\n{'='*70}")
    print(f"{qid} ({q['type']})  {q['question']}")
    print(f"{'='*70}")
    print(f"{'gold 청크':52}{'BM25':>8}{'Dense':>8}")
    for g in gold:
        short = g.split("::", 1)[1] if "::" in g else g
        doc = "초" if "(초)" in g else ("중" if "(중)" in g else
              ("고" if "(고)" in g else "총론"))
        print(f"  [{doc}] {short:45}{bm_rank[g]:>8}{de_rank[g]:>8}")

    print(f"\n  BM25 top5:")
    for i in np.argsort(-bs)[:5]:
        mark = "★" if ids[i] in gold else " "
        print(f"   {mark} {bs[i]:6.2f}  {by_id[ids[i]]['path'][:55]}")

    print(f"  Dense top5:")
    for i in np.argsort(-ds)[:5]:
        mark = "★" if ids[i] in gold else " "
        print(f"   {mark} {ds[i]:6.3f}  {by_id[ids[i]]['path'][:55]}")