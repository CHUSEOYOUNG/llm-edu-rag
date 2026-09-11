"""Small, deterministic query-planning helpers for scope-heavy questions.

The original question remains authoritative.  A supplemental query only removes
legal/date framing that can dominate body-only semantic retrieval.
"""

from __future__ import annotations

import re


DATE = r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일"
NOTICE = (
    r"(?:국가교육위원회|교육부)\s*고시\s*제\d{4}-\d+호"
)
CONTENT_REQUEST = re.compile(
    r"무엇|어떤|목록|종류|교과|과목|영역|내용|기준|방법"
)
SCOPE_HEAVY = re.compile(rf"{DATE}|{NOTICE}")
SCHOOL_NAMES = {
    "초등학교": "elementary",
    "중학교": "middle",
    "고등학교": "high",
}
SCHOOL_COMPARISON = re.compile(
    r"초등학교\s*(?:와|과|및|·|,)\s*중학교"
    r"|중학교\s*(?:와|과|및|·|,)\s*고등학교"
    r"|초등학교\s*(?:와|과|및|·|,)\s*고등학교"
)


def supplemental_content_query(question: str) -> str | None:
    """Return a content-focused query while retaining scope in the caller.

    This is intentionally narrow.  Ordinary questions are not rewritten, and
    the helper never adds words that were not already represented by the input.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("질문은 비어 있을 수 없습니다.")
    if not SCOPE_HEAVY.search(question) or not CONTENT_REQUEST.search(question):
        return None

    result = question
    result = re.sub(r"코퍼스에\s*포함된\s*", "", result)
    result = re.sub(rf"{NOTICE}\s*(?:에서|에\s*따르면)?\s*[,，]?\s*", "", result)
    result = re.sub(
        rf"{DATE}\s*(?:부터|을\s*기준으로|를\s*기준으로|기준으로|기준)?\s*",
        "",
        result,
    )
    result = re.sub(r"적용하도록\s*정한\s*", "", result)
    result = re.sub(r"적용되는\s*|시행되는\s*", "", result)
    result = re.sub(r"\s+", " ", result).strip(" ,，")
    meaningful = [token for token in re.findall(r"[0-9A-Za-z가-힣]+", result)
                  if len(token) >= 2]
    return result if len(meaningful) >= 2 and result != question.strip() else None


def school_comparison_queries(question: str) -> list[dict[str, str]]:
    """Split an explicit multi-school comparison into school-specific searches."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("질문은 비어 있을 수 없습니다.")
    schools = [name for name in SCHOOL_NAMES if name in question]
    if len(schools) < 2 or not SCHOOL_COMPARISON.search(question):
        return []

    results = []
    for target in schools:
        rewritten = question
        for other in schools:
            if other == target:
                continue
            rewritten = re.sub(
                rf"{re.escape(target)}\s*(?:와|과|및|·|,)\s*{re.escape(other)}의?",
                target,
                rewritten,
            )
            rewritten = re.sub(
                rf"{re.escape(other)}\s*(?:와|과|및|·|,)\s*{re.escape(target)}의?",
                target,
                rewritten,
            )
        rewritten = re.sub(
            r"\d{4}\s*개정\s*교육과정\s*기준으로\s*", "교육과정 ", rewritten
        )
        rewritten = re.sub(
            rf"^교육과정\s+{re.escape(target)}\b", f"{target} 교육과정", rewritten
        )
        rewritten = re.sub(r"\b각각\b\s*", "", rewritten)
        rewritten = re.sub(
            r"([0-9A-Za-z가-힣]+)(?:은|는)\s+(?=(?:최소\s+)?몇|어떤)",
            r"\1 ",
            rewritten,
        )
        rewritten = re.sub(r"이?\s*원칙인가요\s*[?？]?\s*$", "", rewritten)
        rewritten = re.sub(r"편성해야\s*하나요\s*[?？]?\s*$", "편성", rewritten)
        rewritten = re.sub(
            r"어떤\s+([0-9A-Za-z가-힣]+)으로\s+구성되나요\s*[?？]?\s*$",
            r"\1 구성",
            rewritten,
        )
        rewritten = re.sub(r"\s+", " ", rewritten).strip()
        if rewritten != question.strip():
            results.append({"school_level": SCHOOL_NAMES[target], "query": rewritten})
    return results


def interleave_rankings(rankings: list[list[dict]], limit: int) -> list[dict]:
    """Round-robin rankings with stable chunk-id deduplication."""
    if limit < 1:
        raise ValueError("결과 개수는 양수여야 합니다.")
    if not rankings or any(not isinstance(ranking, list) for ranking in rankings):
        raise ValueError("하나 이상의 순위 목록이 필요합니다.")
    merged, seen = [], set()
    depth = max((len(ranking) for ranking in rankings), default=0)
    for rank in range(depth):
        for ranking in rankings:
            if rank >= len(ranking):
                continue
            hit = ranking[rank]
            chunk_id = hit.get("chunk_id")
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            merged.append(hit)
            if len(merged) == limit:
                return merged
    return merged
