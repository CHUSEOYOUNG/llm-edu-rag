"""Run retrieval and local-generation regression checks as separate scorecards."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import time
from typing import Any


KS = (1, 5, 10)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def school_level_for(question: dict[str, Any]) -> str:
    value = (question.get("meta") or {}).get("school_level")
    return {"초": "elementary", "중": "middle", "고": "high"}.get(value, "all")


def evidence_groups(question: dict[str, Any]) -> list[set[str]]:
    groups = question.get("evidence_groups") or []
    if groups:
        return [{item["chunk_id"] for item in group["alternatives"]} for group in groups]
    gold = set(question.get("gold_chunks") or [])
    return [gold] if gold else []


def retrieval_metrics(ranked_ids: list[str], groups: list[set[str]]) -> dict[str, float]:
    if not groups or any(not group for group in groups):
        raise ValueError("검색 평가는 하나 이상의 정답 근거 그룹이 필요합니다.")
    union = set().union(*groups)
    metrics: dict[str, float] = {}
    for k in KS:
        top = set(ranked_ids[:k])
        covered = sum(bool(top & group) for group in groups)
        metrics[f"hit@{k}"] = float(bool(top & union))
        metrics[f"recall@{k}"] = covered / len(groups)
        metrics[f"complete@{k}"] = float(covered == len(groups))
    metrics["mrr@10"] = next(
        (1 / rank for rank, chunk_id in enumerate(ranked_ids[:10], 1)
         if chunk_id in union), 0.0
    )
    return metrics


def citation_is_valid(answer: dict[str, Any]) -> bool:
    if answer.get("status") != "draft_answer" or not answer.get("claims"):
        return False
    sources = {source["source_id"]: source for source in answer["context"]["sources"]}
    for claim in answer["claims"]:
        if not claim.get("text") or not claim.get("evidence"):
            return False
        for evidence in claim["evidence"]:
            source = sources.get(evidence.get("source_id"))
            start, end = evidence.get("start"), evidence.get("end")
            if (source is None or evidence.get("chunk_id") != source.get("chunk_id")
                    or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(source["body"])
                    or source["body"][start:end] != evidence.get("quote")):
                return False
    return True


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return round(ordered[index], 1)


def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: round(statistics.fmean(row[key] for row in rows), 4)
            for key in rows[0]}


def rate(rows: list[dict[str, Any]], predicate) -> float | None:
    return round(statistics.fmean(predicate(row) for row in rows), 4) if rows else None


def run_regression(service, questions: list[dict[str, Any]], *, top_k: int = 10,
                   generate: bool = False) -> dict[str, Any]:
    if not 1 <= top_k <= 20:
        raise ValueError("top_k는 1~20이어야 합니다.")
    retrieval_rows, generation_rows = [], []
    for question in questions:
        payload = {
            "question": question["question"],
            "top_k": top_k,
            "school_level": school_level_for(question),
        }
        started = time.perf_counter()
        found = service.search(payload)
        search_ms = round((time.perf_counter() - started) * 1000, 1)
        groups = evidence_groups(question)
        if groups:
            ranked = [source["chunk_id"] for source in found["context"]["sources"]]
            retrieval_rows.append({
                "qid": question["qid"],
                "metrics": retrieval_metrics(ranked, groups),
                "ranked_chunk_ids": ranked,
                "latency_ms": search_ms,
            })
        if not generate:
            continue
        started = time.perf_counter()
        answer = service.answer(payload)
        generation_ms = round((time.perf_counter() - started) * 1000, 1)
        answerable = question["type"] != "unans"
        expected_status = "draft_answer" if answerable else "insufficient_evidence"
        generation_rows.append({
            "qid": question["qid"],
            "answerable": answerable,
            "expected_status": expected_status,
            "actual_status": answer.get("status"),
            "task_success": answer.get("status") == expected_status,
            "citation_ok": citation_is_valid(answer) if answerable else None,
            "latency_ms": generation_ms,
            "answer": answer.get("answer"),
            "reason": answer.get("reason", ""),
        })

    search_latencies = [row["latency_ms"] for row in retrieval_rows]
    result: dict[str, Any] = {
        "schema_version": 1,
        "question_count": len(questions),
        "retrieval": {
            "evaluated_count": len(retrieval_rows),
            "overall": average_metrics([row["metrics"] for row in retrieval_rows]),
            "latency": {
                "p50_ms": percentile(search_latencies, .5),
                "p95_ms": percentile(search_latencies, .95),
            },
            "per_question": retrieval_rows,
        },
        "generation": {"enabled": generate, "per_question": generation_rows},
    }
    if generate:
        generation_latencies = [row["latency_ms"] for row in generation_rows]
        answerable_rows = [row for row in generation_rows if row["answerable"]]
        rejected_rows = [row for row in generation_rows if not row["answerable"]]
        result["generation"].update({
            "overall": {
                "task_success_rate": rate(generation_rows, lambda row: row["task_success"]),
                "answer_success_rate": rate(
                    answerable_rows, lambda row: row["actual_status"] == "draft_answer"),
                "unanswerable_rejection_rate": rate(
                    rejected_rows, lambda row: row["actual_status"] == "insufficient_evidence"),
                "citation_valid_rate": rate(answerable_rows, lambda row: row["citation_ok"]),
            },
            "latency": {
                "p50_ms": percentile(generation_latencies, .5),
                "p95_ms": percentile(generation_latencies, .95),
            },
            "citation_validation": "literal source span only; semantic entailment is not measured",
            "answer_correctness": "not measured; generated answers require manual review",
        })
    return result


def regression_failures(current: dict[str, Any], baseline: dict[str, Any],
                        tolerance: float = 0.0) -> list[str]:
    paths = [
        ("retrieval", "overall", "recall@5"),
        ("retrieval", "overall", "complete@5"),
        ("retrieval", "overall", "mrr@10"),
    ]
    if current["generation"]["enabled"] and baseline["generation"]["enabled"]:
        paths += [
            ("generation", "overall", "task_success_rate"),
            ("generation", "overall", "citation_valid_rate"),
        ]
    failures = []
    for path in paths:
        now = current
        before = baseline
        for key in path:
            now = now[key]
            before = before[key]
        if now + tolerance < before:
            failures.append(f"{'.'.join(path)}: {before:.4f} -> {now:.4f}")
    return failures


def print_scorecard(result: dict[str, Any]) -> None:
    print("\n검색 평가 (검색 근거 회수만 측정)")
    print("qid     R@1    R@5   R@10  Complete@5  MRR@10   ms")
    for row in result["retrieval"]["per_question"]:
        metric = row["metrics"]
        print(f'{row["qid"]:5}  {metric["recall@1"]:5.2f}  {metric["recall@5"]:5.2f}  '
              f'{metric["recall@10"]:5.2f}       {metric["complete@5"]:5.2f}    '
              f'{metric["mrr@10"]:5.2f}  {row["latency_ms"]:5.0f}')
    print("overall", json.dumps(result["retrieval"]["overall"], ensure_ascii=False))

    if not result["generation"]["enabled"]:
        print("\n생성 평가는 --generate를 지정할 때만 실행합니다.")
        return
    print("\n생성 평가 (검색 점수와 별도)")
    print("qid    expected                 actual                   citation    ms")
    for row in result["generation"]["per_question"]:
        citation = "-" if row["citation_ok"] is None else ("ok" if row["citation_ok"] else "fail")
        print(f'{row["qid"]:5}  {row["expected_status"]:23}  '
              f'{row["actual_status"]:23}  {citation:8} {row["latency_ms"]:7.0f}')
    print("overall", json.dumps(result["generation"]["overall"], ensure_ascii=False))
    print("주의: 성공률은 상태와 원문 인용 형식만 측정하며 답변 의미의 정확성은 수동 검토 대상입니다.")
