import json
import pathlib
import unicodedata
import pymupdf4llm

from normalize import normalize, count_stats, strip_running_heads

RAW = pathlib.Path("data/raw")
OUT = pathlib.Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)


def nfc(s: str) -> str:
    """맥 파일명 자모분리(NFD) → 정상 한글(NFC)"""
    return unicodedata.normalize("NFC", s)


pdfs = sorted(RAW.glob("*.pdf"))
print(f"대상 PDF: {len(pdfs)}개\n")

results = []
for pdf in pdfs:
    name = nfc(pdf.name)
    print(name)

    try:
        pages = pymupdf4llm.to_markdown(
            str(pdf), page_chunks=True, show_progress=False
        )
    except Exception as e:
        print(f"  [실패] {e}\n")
        continue

    page_texts = strip_running_heads([p["text"] for p in pages])
    md = "\n\n".join(page_texts)

    raw_stats = count_stats(md)
    md = normalize(md)
    stats = count_stats(md)

    results.append({
        "doc_id": nfc(pdf.stem),
        "source": name,
        "n_pages": len(pages),
        "text": md,
        **stats,
    })

    print(f"  페이지 {len(pages)} / 글자 {stats['n_chars']:,} / 헤딩 {stats['n_heading']} / 표 {stats['n_table']}")
    print(f"  고아 가운뎃점 {raw_stats['n_orphan_dot']} → {stats['n_orphan_dot']}\n")

with open(OUT / "docs.jsonl", "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

sample = OUT / "sample"
sample.mkdir(exist_ok=True)
for r in results:
    (sample / f"{r['doc_id']}.md").write_text(r["text"], encoding="utf-8")

print(f"완료. {len(results)}개 저장 → {OUT/'docs.jsonl'}")