import re
import unicodedata

MIDDLE_DOTS = "\u22c5\uff65\u00b7\u2027\u30fb"  # ⋅ ･ · ‧ ・
_MARKER = re.compile(r"^[\s\-*>]+")
_ONLY_DOTS = re.compile(rf"^[\s{MIDDLE_DOTS}]+$")


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
            continue                                    # 고아 점 줄 삭제
        line = re.sub(rf"^[\s\-*]*[{MIDDLE_DOTS}]+\s+", "", line)  # 본문 앞 점 제거
        out.append(line)
    text = "\n".join(out)

    text = re.sub(rf"[{MIDDLE_DOTS}]", "·", text)       # 문자 통일
    text = re.sub(r"[·．.]{4,}", " ", text)             # 목차 점선
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_stats(text: str) -> dict:
    lines = text.splitlines()
    return {
        "n_chars": len(text),
        "n_heading": sum(1 for l in lines if l.startswith("#")),
        "n_table": text.count("|---"),
        "n_orphan_dot": sum(1 for l in lines if _is_orphan_dot_line(l)),
    }