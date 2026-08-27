"""strip_running_heads 진단."""

import json
import pathlib
from collections import Counter

import normalize as N

CACHE = pathlib.Path("data/processed/raw_pages.json")
pages_by_doc = json.loads(CACHE.read_text(encoding="utf-8"))

doc = list(pages_by_doc)[0]
pl = pages_by_doc[doc]

print(f"문서: {doc[:50]}")
print(f"페이지 {len(pl)}개 / 제거 전 {sum(len(p) for p in pl):,}자\n")

print("--- 원본 1페이지 앞 5줄 ---")
for i, line in enumerate(pl[0].splitlines()[:5]):
    print(f"  [{i}] {line[:70]!r}")

# 반복 패턴 확인
counter = Counter()
for p in pl:
    ls = p.splitlines()
    idx = [i for i, l in enumerate(ls) if l.strip()]
    for i in set(idx[:3] + idx[-3:]):
        k = N._key(ls[i])
        if k and len(k) <= 60:
            counter[k] += 1

print("\n--- 반복 3회 이상 패턴 상위 15개 ---")
for k, c in counter.most_common(15):
    print(f"  {c:4}회  {k[:60]!r}")

print()
out = N.strip_running_heads(list(pl))
print(f"제거 후 {sum(len(p) for p in out):,}자 / 페이지 {len(out)}개")

print("\n--- 제거 후 1페이지 앞 5줄 ---")
if not out or not out[0].strip():
    print("  (비어 있음)")
else:
    for i, line in enumerate(out[0].splitlines()[:5]):
        print(f"  [{i}] {line[:70]!r}")