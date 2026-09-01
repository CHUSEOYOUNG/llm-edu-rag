import json
import pathlib
import re

IN = pathlib.Path("data/processed/sections.jsonl")
OUT = pathlib.Path("data/processed/chunks.jsonl")

TARGET = 800     # 목표 크기 (자)
MAX = 1500       # 하드 리밋
MIN = 100        # 이보다 작으면 다음 청크에 흡수

_TABLE = re.compile(r"^\s*\|")


def split_units(text: str, line_pages: list[int] | None = None) -> list:
    """표 블록은 통째로, 나머지는 문단 단위로 분리."""
    units, buf, page_buf, in_table = [], [], [], False

    lines = text.splitlines()
    if line_pages is not None and len(line_pages) != len(lines):
        raise ValueError("섹션 본문과 페이지 정보의 줄 수가 다릅니다.")

    for index, line in enumerate(lines):
        page = line_pages[index] if line_pages is not None else None
        is_tbl = bool(_TABLE.match(line))

        if is_tbl != in_table:              # 표 <-> 본문 전환
            if buf:
                units.append(("table" if in_table else "text", "\n".join(buf), set(page_buf)))
            buf, page_buf, in_table = [], [], is_tbl

        if not is_tbl and not line.strip() and buf:
            units.append(("text", "\n".join(buf), set(page_buf)))
            buf, page_buf = [], []
            continue

        buf.append(line)
        if page is not None:
            page_buf.append(page)

    if buf:
        units.append(("table" if in_table else "text", "\n".join(buf), set(page_buf)))

    return [(k, t.strip(), pages) for k, t, pages in units if t.strip()]


def hard_split(text: str, size: int) -> list:
    """문장 경계로 강제 분할."""
    parts = re.split(r"(?<=[.。다\.])\s+", text)
    out, buf = [], ""
    for p in parts:
        if len(buf) + len(p) > size and buf:
            out.append(buf.strip())
            buf = ""
        buf += p + " "
    if buf.strip():
        out.append(buf.strip())
    return out


def chunk_section(sec: dict, sec_idx: int) -> list:
    body = sec["text"].strip()
    if not body:
        return []

    header = " > ".join(filter(None, [
        sec["path"],
        f"{sec['number']} {sec['title']}".strip(),
    ]))

    units = split_units(body, sec.get("line_pages"))
    groups, buf, page_buf = [], [], set()

    for kind, text, pages in units:
        if len(text) > MAX:
            if buf:
                groups.append(("\n\n".join(buf), set(page_buf)))
                buf, page_buf = [], set()
            pieces = [text] if kind == "table" else hard_split(text, TARGET)
            groups.extend((piece, set(pages)) for piece in pieces)
            continue

        cur = sum(len(b) for b in buf)
        if cur + len(text) > TARGET and buf:
            groups.append(("\n\n".join(buf), set(page_buf)))
            buf, page_buf = [], set()
        buf.append(text)
        page_buf.update(pages)

    if buf:
        groups.append(("\n\n".join(buf), set(page_buf)))

    # 너무 작은 마지막 조각은 앞에 흡수
    if len(groups) > 1 and len(groups[-1][0]) < MIN:
        tail, tail_pages = groups.pop()
        previous, previous_pages = groups[-1]
        groups[-1] = (previous + "\n\n" + tail, previous_pages | tail_pages)

    chunks = [{
        "chunk_id": f"{sec['doc_id']}::s{sec_idx:03d}::{sec['number']}::{i}",
        "doc_id": sec["doc_id"],
        "path": header,
        "text": f"{header}\n\n{g}" if header else g,
        "body": g,
        "n_chars": len(g),
        "part": i,
        "n_parts": len(groups),
        **({"page_start": min(pages), "page_end": max(pages)} if pages else {}),
    } for i, (g, pages) in enumerate(groups)]
    return chunks


def main():
    sections = [json.loads(l) for l in IN.open(encoding="utf-8")]
    chunks = []
    for idx, s in enumerate(sections):
        chunks.extend(chunk_section(s, idx))

    chunks = [c for c in chunks if c["n_chars"] >= 30]

    with OUT.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    sizes = sorted(c["n_chars"] for c in chunks)
    n = len(sizes)
    print(f"섹션 {len(sections)} → 청크 {n}")
    print(f"  최소 {sizes[0]:,} / 25% {sizes[n//4]:,} / 중앙 {sizes[n//2]:,} "
          f"/ 75% {sizes[3*n//4]:,} / 최대 {sizes[-1]:,}")
    print(f"  1500자 초과: {sum(1 for s in sizes if s > MAX)}개")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
