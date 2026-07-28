import re
import unicodedata
from collections import Counter

MIDDLE_DOTS = "\u22c5\uff65\u00b7\u2027\u30fb"  # ⋅ ･ · ‧ ・
_MARKER = re.compile(r"^[\s\-*>]+")
_ONLY_DOTS = re.compile(rf"^[\s{MIDDLE_DOTS}]+$")
_PAGENUM = re.compile(r"^[\s\-–—]*\d{1,4}[\s\-–—]*$")


# ---------- 가운뎃점 정규화 ----------

def _is_orphan_dot_line(line: str) -> bool:
    """리스트 마커를 걷어낸 뒤 가운뎃점만 남는 줄인가"""
    body = _MARKER.sub("", line)
    return bool(body.strip()) and bool(_ONLY_DOTS.match(body))


def normalize(text: str) -> str:
    """파싱 산출물 1차 정규화."""
    text = unicodedata.normalize("NFC", text)

    out = []
    for line in text.splitlines():
        if _is_orphan_dot_line(line):
            continue                                                    # 고아 점 줄 삭제
        line = re.sub(rf"^[\s\-*]*[{MIDDLE_DOTS}]+\s+", "", line)       # 본문 앞 점 제거
        out.append(line)
    text = "\n".join(out)

    text = re.sub(rf"[{MIDDLE_DOTS}]", "·", text)       # 가운뎃점 문자 통일
    text = re.sub(r"[·．.]{4,}", " ", text)             # 목차 점선 제거
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)     # 줄 끝 공백
    text = re.sub(r"\n{3,}", "\n\n", text)              # 빈 줄 압축
    return text.strip()


# ---------- 머리말/꼬리말 제거 ----------

def _key(line: str) -> str:
    """비교용 정규화: 마크다운 마커 제거"""
    s = re.sub(r"^#{1,6}\s*", "", line.strip())
    s = re.sub(r"[*_`]", "", s)
    return s.strip()

def strip_running_heads(pages: list, edge: int = 3, min_repeat: int = 3) -> list:
    """페이지 가장자리에서 반복되는 머리말/꼬리말/페이지번호 제거.

    반복 패턴은 첫 등장만 남기고 이후 것을 제거한다.
    (진짜 제목 1개 + 머리말 N개인 경우를 보존하기 위함)
    """
    n = len(pages)
    if n < 5:
        return pages

    def edge_indices(lines):
        idx = [i for i, l in enumerate(lines) if l.strip()]
        return set(idx[:edge] + idx[-edge:])

    counter = Counter()
    for p in pages:
        lines = p.splitlines()
        for i in edge_indices(lines):
            k = _key(lines[i])
            if k and len(k) <= 60:
                counter[k] += 1

    repeated = {k for k, c in counter.items() if c >= min_repeat}

    seen = set()
    out, removed = [], 0
    for p in pages:
        lines = p.splitlines()
        ei = edge_indices(lines)
        kept = []
        for i, l in enumerate(lines):
            if i in ei:
                k = _key(l)
                if _PAGENUM.match(k):
                    removed += 1
                    continue
                if k in repeated:
                    if k in seen:          # 첫 등장이 아니면 제거
                        removed += 1
                        continue
                    seen.add(k)
            kept.append(l)
        out.append("\n".join(kept))

    print(f"  머리말/꼬리말 {removed}줄 제거 (반복 패턴 {len(repeated)}종)")
    return out


# ---------- 통계 ----------

def count_stats(text: str) -> dict:
    lines = text.splitlines()
    return {
        "n_chars": len(text),
        "n_heading": sum(1 for l in lines if l.startswith("#")),
        "n_table": text.count("|---"),
        "n_orphan_dot": sum(1 for l in lines if _is_orphan_dot_line(l)),
    }