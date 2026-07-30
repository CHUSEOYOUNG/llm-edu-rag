import json
import pathlib
from collections import Counter

chunks = {json.loads(l)["chunk_id"]
          for l in pathlib.Path("data/processed/chunks.jsonl").open(encoding="utf-8")}
qs = [json.loads(l)
      for l in pathlib.Path("eval/questions.jsonl").open(encoding="utf-8")]

print(f"질문 {len(qs)}개 | 유형 {dict(Counter(q['type'] for q in qs))}\n")

bad = 0
for q in qs:
    g = q.get("gold_chunks") or []
    if q["type"] == "unans":
        if g:
            print(f"  [경고] {q['qid']}: unans인데 gold 있음")
            bad += 1
        continue
    if not g:
        print(f"  [미완] {q['qid']}: gold 없음")
        bad += 1
        continue
    missing = [c for c in g if c not in chunks]
    if missing:
        print(f"  [오류] {q['qid']}: 존재하지 않는 청크 {missing}")
        bad += 1

sizes = [len(q.get('gold_chunks') or []) for q in qs if q['type'] != 'unans']
if sizes:
    print(f"\ngold 청크 수: 평균 {sum(sizes)/len(sizes):.1f} / 최대 {max(sizes)}")
print(f"\n문제 {bad}건" if bad else "\n이상 없음")