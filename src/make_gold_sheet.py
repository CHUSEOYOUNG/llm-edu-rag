"""질문별 후보 청크를 마크다운으로 출력. 파일에서 골라 표시한 뒤 apply로 반영."""
import json
import pathlib
from find_gold import bigrams, search

CHUNKS = pathlib.Path("data/processed/chunks.jsonl")
QUESTIONS = pathlib.Path("eval/questions.jsonl")
OUT = pathlib.Path("eval/gold_sheet.md")

chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
indexed = [bigrams(c["path"] + " " + c["body"]) for c in chunks]
qs = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8")]

lines = ["# gold 선택 시트", "",
         "정답 청크의 `[ ]`를 `[x]`로 바꾸고 저장 → `uv run python src/apply_gold.py`", ""]

for q in qs:
    if q["type"] == "unans":
        continue
    lines.append(f"\n## {q['qid']} ({q['type']})")
    lines.append(f"**{q['question']}**")
    lines.append(f"> 정답: {q.get('answer')}")
    hint = (q.get("meta") or {}).get("source_hint")
    if hint:
        lines.append(f"> 힌트: {hint}")
    lines.append("")

    for score, c in search(q["question"] + " " + (q.get("answer") or ""),
                           chunks, indexed, n=12):
        mark = "x" if c["chunk_id"] in (q.get("gold_chunks") or []) else " "
        body = c["body"][:150].replace("\n", " ")
        lines.append(f"- [{mark}] `{c['chunk_id']}` ({score:.2f})")
        lines.append(f"      {c['path'][:60]}")
        lines.append(f"      {body}")
    lines.append("")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"→ {OUT}")