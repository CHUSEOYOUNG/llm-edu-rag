"""전처리 단계별 기여도 측정 (ablation).

chunk_id는 전처리 설정에 따라 달라지므로,
gold 판정을 '정답 문자열 포함 여부'로 수행한다.
"""

import contextlib
import io
import json
import pathlib
import re
import unicodedata

import numpy as np
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

import normalize as N
from sections import extract_sections, merge_orphans
from chunk import chunk_section

RAW_DIR = pathlib.Path("data/raw")
CACHE = pathlib.Path("data/processed/raw_pages.json")
QUESTIONS = pathlib.Path("eval/questions.jsonl")
OUT = pathlib.Path("experiments/ablation_preprocess.json")

KS = [1, 5, 10]
KEEP_POS = {"NNG", "NNP", "NNB", "NR", "SL", "SN", "SH", "VV", "VA", "MAG"}
kiwi = Kiwi()

# qid: 정답 청크라면 포함해야 할 문자열 (하나라도 있으면 gold)
GOLD_TEXT = {
    "q001": ["제4호 제5호의 조치사항은 학생의 졸업일로부터 2년",
             "제4호 및 제5호의 조치사항"],
    "q002": ["객관적인 증빙자료가 있는 경우에만 정정이 가능"],
    "q003": ["한글 1자는 3Byte"],
    "q004": ["20% 범위 내에서 시수를 증감"],
    "q005": ["선택 교과는 한문"],
    "q006": ["수업은 40분", "수업은 45분"],
    "q007": ["34시간 이상", "68시간 이상"],
    "q008": ["바른 생활, 슬기로운 생활", "바른 생활 슬기로운 생활"],
    "q009": ["동아리 활동, 진로 활동"],
    "q010": ["국어|442"],
    "q011": ["|408|"],
}


def tokenize(t):
    return [x.form for x in kiwi.tokenize(t) if x.tag in KEEP_POS]


def load_raw_pages():
    """PDF → 페이지별 마크다운. 캐시 사용."""
    if CACHE.exists():
        print(f"캐시 사용: {CACHE}")
        return json.loads(CACHE.read_text(encoding="utf-8"))

    import pymupdf4llm
    print("PDF 파싱 중 (최초 1회, 수 분 소요)...")
    data = {}
    for pdf in sorted(RAW_DIR.glob("*.pdf")):
        doc_id = unicodedata.normalize("NFC", pdf.stem)
        pages = pymupdf4llm.to_markdown(
            str(pdf), page_chunks=True, show_progress=False)
        data[doc_id] = [p["text"] for p in pages]
        print(f"  {doc_id[:40]}: {len(pages)}p")
    CACHE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def minimal_norm(text: str) -> str:
    """정규화 없음 조건: NFC + 공백 정리만"""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def build_chunks(raw_pages, use_dot, use_head):
    chunks = []
    for doc_id, pages in raw_pages.items():
        page_list = list(pages)
        if use_head:
            with contextlib.redirect_stdout(io.StringIO()):
                page_list = N.strip_running_heads(page_list)

        md = "\n\n".join(page_list)
        md = N.normalize(md) if use_dot else minimal_norm(md)

        secs = extract_sections(doc_id, md)
        secs = [s for s in secs if s.kind != "appendix"]
        secs, _ = merge_orphans(secs)
        for i, s in enumerate(secs):
            chunks.extend(chunk_section(s.to_dict(), i))

    return [c for c in chunks if c["n_chars"] >= 30]


def eval_bm25(chunks, questions):
    bodies = [c["path"] + " " + c["body"] for c in chunks]
    bm25 = BM25Okapi([tokenize(b) for b in bodies])

    res, skipped = [], []
    for q in questions:
        pats = GOLD_TEXT.get(q["qid"])
        if not pats:
            skipped.append(q["qid"])
            continue

        gold_idx = {i for i, b in enumerate(bodies)
                    if any(p in b for p in pats)}
        if not gold_idx:
            skipped.append(q["qid"])
            continue

        scores = bm25.get_scores(tokenize(q["question"]))
        ranked = list(np.argsort(-scores)[:20])

        m = {"qid": q["qid"], "type": q["type"], "n_gold": len(gold_idx)}
        for k in KS:
            m[f"recall@{k}"] = len(gold_idx & set(ranked[:k])) / len(gold_idx)
        m["mrr"] = next((1 / i for i, x in enumerate(ranked, 1)
                         if x in gold_idx), 0.0)
        res.append(m)

    return res, skipped


def main():
    raw_pages = load_raw_pages()
    questions = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8")]
    questions = [q for q in questions if q["type"] != "unans"]

    configs = [
        ("raw",  False, False),
        ("dot",  True,  False),
        ("head", False, True),
        ("both", True,  True),
    ]

    results = {}
    for name, use_dot, use_head in configs:
        print(f"\n[{name}] 가운뎃점={use_dot} 머리말={use_head}")
        chunks = build_chunks(raw_pages, use_dot, use_head)
        print(f"  청크 {len(chunks)}개")
        if not chunks:
            print("  청크 없음 — 건너뜀")
            continue
        res, skipped = eval_bm25(chunks, questions)
        if skipped:
            print(f"  ※ gold 문자열 미검출: {', '.join(skipped)}")
        results[name] = {"n_chunks": len(chunks), "n_eval": len(res),
                         "skipped": skipped, "per_question": res}

    print(f"\n{'='*64}\n전처리 Ablation (BM25)\n{'='*64}")
    print(f"{'설정':8}{'청크':>8}{'평가':>6}{'R@1':>9}{'R@5':>9}{'R@10':>9}{'MRR':>9}")
    for name, _, _ in configs:
        if name not in results:
            continue
        r = results[name]
        rs = r["per_question"]
        if not rs:
            continue

        def a(k):
            return sum(x[k] for x in rs) / len(rs)

        print(f"{name:8}{r['n_chunks']:>8}{len(rs):>6}"
              f"{a('recall@1'):>9.3f}{a('recall@5'):>9.3f}"
              f"{a('recall@10'):>9.3f}{a('mrr'):>9.3f}")

    # 질문별 비교
    print(f"\n{'='*64}\n질문별 R@5\n{'='*64}")
    names = [n for n, _, _ in configs if n in results]
    print(f"{'qid':8}" + "".join(f"{n:>10}" for n in names))
    for q in questions:
        row = f"{q['qid']:8}"
        found = False
        for n in names:
            hit = next((x for x in results[n]["per_question"]
                        if x["qid"] == q["qid"]), None)
            row += f"{hit['recall@5']:>10.2f}" if hit else f"{'-':>10}"
            found = found or hit is not None
        if found:
            print(row)

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()