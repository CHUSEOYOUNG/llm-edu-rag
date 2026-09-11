"""Extract narrow, question-linked facts from damaged Markdown tables."""

from __future__ import annotations

import re


GRADE_RANGE = re.compile(
    r"(?P<start>\d+)\s*[·~∼〜-]\s*(?P<end>\d+)\s*학년"
)
QUANTITY_UNIT = re.compile(
    r"몇\s*(?P<unit>분|시간|일|주|개월|달|년|회|번|개|명|쪽|학점)"
)
PERIOD_NOTE = re.compile(
    r"연간\s*(?P<weeks>\d+)\s*주.{0,60}?"
    r"(?P<years>[2-9]\d*)\s*년간의\s*기준\s*수업\s*시수",
    re.DOTALL,
)
TABLE_SEPARATOR = re.compile(r"^\s*:?-{3,}:?\s*$")
QUESTION_STOPWORDS = {
    "초등학교", "중학교", "고등학교", "학년", "수업", "시수", "시간",
    "몇", "인가요", "알려줘", "기준", "교과", "과목",
}


def readable(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def table_cells(line: str) -> list[str]:
    return [readable(cell) for cell in line.strip().strip("|").split("|")]


def normalized_grade(text: str) -> str | None:
    match = GRADE_RANGE.search(text)
    if not match:
        return None
    return f"{match.group('start')}~{match.group('end')}학년"


def contains_numeric_value(text: str, value: str) -> bool:
    """Match a table number without accepting it as part of a larger number."""
    digits = value.replace(",", "")
    normalized = text.replace(",", "")
    return bool(re.search(rf"(?<!\d){re.escape(digits)}(?!\d)", normalized))


def question_terms(question: str) -> list[str]:
    terms = {
        term for term in re.findall(r"[가-힣]{2,}", question)
        if term not in QUESTION_STOPWORDS and not term.endswith("학년")
    }
    return sorted(terms, key=lambda term: (-len(term), term))


def table_blocks(body: str) -> list[str]:
    return [
        block.strip() for block in re.split(r"\n\s*\n", body)
        if sum(line.strip().startswith("|") for line in block.splitlines()) >= 2
    ]


def period_details(body: str) -> tuple[str | None, str | None, str | None]:
    for block in re.split(r"\n\s*\n", body):
        match = PERIOD_NOTE.search(readable(block))
        if match:
            return (
                f"{match.group('years')}년간",
                f"연간 {match.group('weeks')}주 기준",
                block.strip(),
            )
    return None, None, None


def extract_table_facts(question: str, sources: list[dict]) -> list[dict]:
    """Return only cells whose row and grade column are both named in the question."""
    target_grade = normalized_grade(question)
    unit_match = QUANTITY_UNIT.search(question)
    terms = question_terms(question)
    if target_grade is None or unit_match is None or not terms:
        return []

    facts = []
    for source in sources:
        period, basis, period_quote = period_details(source["body"])
        candidates = []
        for block in table_blocks(source["body"]):
            lines = [line.strip() for line in block.splitlines()
                     if line.strip().startswith("|")]
            rows = [table_cells(line) for line in lines]
            header_index = next((index for index, row in enumerate(rows)
                                 if target_grade in {
                                     normalized_grade(cell) for cell in row
                                 }), None)
            if header_index is None:
                continue
            header = rows[header_index]
            column = next(index for index, cell in enumerate(header)
                          if normalized_grade(cell) == target_grade)
            for raw_line, row in zip(lines[header_index + 1:], rows[header_index + 1:]):
                if all(TABLE_SEPARATOR.fullmatch(cell) for cell in row if cell):
                    continue
                label_area = " ".join(row[:max(column, 1)])
                matching = [term for term in terms if term in label_area]
                if not matching or column >= len(row):
                    continue
                value_cell = row[column]
                numbers = re.findall(r"(?<!\d)\d[\d,]*(?!\d)", value_cell)
                if column == 0:
                    numbers = re.findall(r"(?<!\d)\d[\d,]*(?!\d)", row[0])
                if not numbers:
                    continue
                candidates.append((len(matching[0]), {
                    "source_id": source["source_id"],
                    "row_label": matching[0],
                    "column_label": target_grade,
                    "value": numbers[-1],
                    "unit": unit_match.group("unit"),
                    "period": period,
                    "basis": basis,
                    "value_quote": raw_line,
                    "period_quote": period_quote,
                }))
        if candidates:
            candidates.sort(key=lambda item: -item[0])
            facts.append(candidates[0][1])
    return facts
