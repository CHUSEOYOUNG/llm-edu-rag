import re
from dataclasses import dataclass, field

ROMAN = "ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ"
KOR = "가나다라마바사아자차카타파하"
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"

# (유형, 정규식, 레벨) — 레벨이 낮을수록 상위
PATTERNS = [
    ("appendix", re.compile(r"^부\s*록\s*(\d*)"), 1),
    ("chapter", re.compile(r"^제\s*(\d+)\s*장\b"), 1),
    ("roman",   re.compile(rf"^([{ROMAN}]+)\s*[.．]"), 1),
    ("article", re.compile(r"^제\s*(\d+조(?:의\s*\d+)?)\s*[(（]([^)）]*)[)）]"), 2),
    ("num_dot", re.compile(r"^(\d+)\s*[.．](?!\d)"), 3),
    ("kor_dot", re.compile(rf"^([{KOR}])\s*[.．]"), 4),
    ("num_par", re.compile(r"^(\d+)\s*[)）]"), 5),
    ("kor_par", re.compile(rf"^([{KOR}])\s*[)）]"), 5),
    ("circled", re.compile(rf"^([{CIRCLED}])"), 6),
]

_HEAD = re.compile(r"^(#{1,6})\s*")
_LIST = re.compile(r"^[-*+>]\s*")
_EMPH = re.compile(r"\*\*|__|\*|`")

# 문장으로 끝나면 제목이 아님
_SENTENCE_END = ("다.", "다", "함.", "임.", "음.", "된다.", "한다.")


def clean_line(line: str):
    """마크다운 마커 제거. (본문, 헤딩여부) 반환"""
    s = line.strip()
    is_head = bool(_HEAD.match(s))
    s = _HEAD.sub("", s)
    s = _LIST.sub("", s)
    s = _EMPH.sub("", s)
    return s.strip(), is_head


def match_number(text: str):
    """번호 체계 매칭 → (유형, 레벨, 번호, 나머지텍스트)"""
    for kind, pat, level in PATTERNS:
        m = pat.match(text)
        if not m:
            continue
        rest = text[m.end():].strip()
        if kind == "article":
            return kind, level, m.group(1), m.group(2), rest
        return kind, level, m.group(1), rest, ""
    return None


def looks_like_title(title: str, is_head: bool) -> bool:
    """제목처럼 보이는가 (본문 속 번호 목록 걸러내기)"""
    if is_head:
        return True
    if not title or len(title) > 50:
        return False
    if title.endswith(_SENTENCE_END):
        return False
    return True


@dataclass
class Section:
    doc_id: str
    level: int
    kind: str
    number: str
    title: str
    path: list = field(default_factory=list)
    lines: list = field(default_factory=list)

    def to_dict(self):
        body = "\n".join(self.lines).strip()
        return {
            "doc_id": self.doc_id,
            "level": self.level,
            "kind": self.kind,
            "number": self.number,
            "title": self.title,
            "path": " > ".join(self.path),
            "text": body,
            "n_chars": len(body),
        }


def extract_sections(doc_id: str, text: str) -> list:
    sections = []
    stack = []  # (level, title)
    current = Section(doc_id, 0, "root", "", "(문서 서두)")

    for line in text.splitlines():
        clean, is_head = clean_line(line)
        if not clean:
            current.lines.append("")
            continue

        hit = match_number(clean)
        if hit:
            kind, level, number, title, rest = hit
            if looks_like_title(title, is_head):
                sections.append(current)

                while stack and stack[-1][0] >= level:
                    stack.pop()
                path = [t for _, t in stack]
                stack.append((level, f"{number} {title}".strip()))

                current = Section(doc_id, level, kind, number, title, path)
                if rest:
                    current.lines.append(rest)
                continue

        current.lines.append(clean)

    sections.append(current)
    return [s for s in sections if s.lines or s.title]

def merge_orphans(sections: list, max_pass: int = 5) -> tuple:
    """본문도 자식도 없는 섹션 = 열거 항목. 부모 본문으로 되돌린다."""
    merged_total = 0

    for _ in range(max_pass):
        n = len(sections)
        has_child = [False] * n
        for i in range(n - 1):
            if sections[i + 1].level > sections[i].level:
                has_child[i] = True

        kept, merged = [], 0
        for i, s in enumerate(sections):
            body = "\n".join(s.lines).strip()
            is_orphan = (not body) and (not has_child[i]) and s.level > 0

            if is_orphan and kept:
                label = f"{s.number} {s.title}".strip()
                kept[-1].lines.append(label)
                merged += 1
                continue
            kept.append(s)

        sections = kept
        merged_total += merged
        if merged == 0:
            break

    return sections, merged_total