import json
import pathlib
import re

SHEET = pathlib.Path("eval/gold_sheet.md")
QUESTIONS = pathlib.Path("eval/questions.jsonl")

qs = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8")]
by_qid = {q["qid"]: q for q in qs}

cur = None
picked = {}
for line in SHEET.read_text(encoding="utf-8").splitlines():
    m = re.match(r"^##\s+(q\d+\w*)", line)
    if m:
        cur = m.group(1)
        picked[cur] = []
        continue
    m = re.match(r"^-\s*\[x\]\s*`([^`]+)`", line, re.I)
    if m and cur:
        picked[cur].append(m.group(1))

for qid, ids in picked.items():
    if qid in by_qid:
        by_qid[qid]["gold_chunks"] = ids
        print(f"{qid}: {len(ids)}개")

with QUESTIONS.open("w", encoding="utf-8") as f:
    for q in qs:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")
print("반영 완료")