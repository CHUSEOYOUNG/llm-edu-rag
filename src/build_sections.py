import json
import pathlib
from collections import Counter

from sections import extract_sections, merge_orphans

IN = pathlib.Path("data/processed/docs.jsonl")
OUT = pathlib.Path("data/processed/sections.jsonl")


def main():
    docs = [json.loads(l) for l in IN.open(encoding="utf-8")]
    all_sections = []

    for d in docs:
        secs = extract_sections(d["doc_id"], d["text"], d.get("page_starts"))
        secs = [s for s in secs if s.kind != "appendix"]
        before = len(secs)
        secs, merged = merge_orphans(secs)
        all_sections.extend(secs)

        # 자식 유무 판정: 다음 섹션의 레벨이 더 깊으면 부모
        has_child = [False] * len(secs)
        for i in range(len(secs) - 1):
            if secs[i + 1].level > secs[i].level:
                has_child[i] = True

        sizes = [len("\n".join(s.lines).strip()) for s in secs]
        orphan = sum(1 for i, n in enumerate(sizes) if n == 0 and not has_child[i])
        parent = sum(1 for i, n in enumerate(sizes) if n == 0 and has_child[i])
        body = sorted(n for n in sizes if n > 0)

        kinds = Counter(s.kind for s in secs)
        print(f"\n{d['source'][:45]}")
        print(f"  섹션 {len(secs)}개 | {dict(kinds)}")
        print(f"  구조노드 {parent} / 고아 빈섹션 {orphan} ({orphan/len(secs):.0%})")
        print(f"  열거항목 병합 {merged}개 ({before} → {len(secs)})")
        if body:
            print(f"  본문 중앙값 {body[len(body)//2]:,} / 최대 {max(body):,}")

    with OUT.open("w", encoding="utf-8") as f:
        for s in all_sections:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

    print(f"\n총 {len(all_sections)}개 섹션 → {OUT}")


if __name__ == "__main__":
    main()
