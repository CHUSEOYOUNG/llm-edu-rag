"""평가셋 확장용 후보 청크 샘플링.

유형별로 적합한 청크를 뽑아 질문 작성 초안을 만든다.
"""

import json
import pathlib
import random
import re

CHUNKS = pathlib.Path("data/processed/chunks.jsonl")
QUESTIONS = pathlib.Path("eval/questions.jsonl")
OUT = pathlib.Path("eval/draft_v2.md")

random.seed(7)

chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
used = set()
for l in QUESTIONS.open(encoding="utf-8"):
    used.update(json.loads(l).get("gold_chunks") or [])

pool = [c for c in chunks if c["chunk_id"] not in used and c["n_chars"] >= 200]

_NUM = re.compile(r"\|\s*[\d,]{2,}\s*\|")


def is_table(c):
    return c["body"].count("|---") >= 1 and len(_NUM.findall(c["body"])) >= 3


def is_prose(c):
    return c["body"].count("|") < c["n_chars"] / 40


tables = [c for c in pool if is_table(c)]
prose = [c for c in pool if is_prose(c)]

# 학교급별로 같은 조항이 있는 것 = cond 후보
by_key = {}
for c in prose:
    if "기재요령" not in c["doc_id"]:
        continue
    key = (c["path"], c["part"])
    by_key.setdefault(key, []).append(c)
cond_pairs = [v for v in by_key.values() if len(v) >= 2]

picks = {
    "fact": random.sample(prose, min(12, len(prose))),
    "table": random.sample(tables, min(10, len(tables))),
    "cond": [random.choice(g) for g in random.sample(cond_pairs, min(8, len(cond_pairs)))],
}

lines = [
    "# 평가셋 확장 초안 v2", "",
    "각 청크를 읽고 질문을 작성. **문서 표현을 그대로 쓰지 말 것.**", "",
    "목표: fact +5 / cond +4 / table +4 / multi +4 / unans +3", "",
    "---", "",
]

for kind, cs in picks.items():
    lines.append(f"\n# [{kind}] 후보 {len(cs)}개\n")
    for c in cs:
        lines.append(f"## `{c['chunk_id']}`")
        lines.append(f"**경로**: {c['path'][:70]}")
        lines.append(f"```\n{c['body'][:500]}\n```")
        lines.append("- 질문: \n- 정답: \n- 사용: [ ]\n")

lines.append("\n# [multi] 직접 설계\n")
lines.append("여러 청크를 종합해야 답이 되는 질문. 예:")
lines.append("- 초/중/고 기재요령을 비교해야 하는 질문")
lines.append("- 조문 + 해설을 함께 봐야 하는 질문")
lines.append("- 총론 + 기재요령을 연결해야 하는 질문\n")

lines.append("\n# [unans] 직접 설계\n")
lines.append("교육 도메인이지만 이 코퍼스에 없는 질문. 예:")
lines.append("- 입시 전형, 교원 임용, 대학 학사, 유치원 관련\n")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"→ {OUT}")
print(f"  표 후보 {len(tables)} / 산문 후보 {len(prose)} / cond 쌍 {len(cond_pairs)}")