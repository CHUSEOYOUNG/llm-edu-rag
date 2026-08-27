"""Run frozen development questions through the RAG CLI components.

Default is real retrieval only. --generate explicitly enables paid API calls.
Gold is used after retrieval for diagnostics and is never sent to the generator.
"""

import argparse
import json
import os
from pathlib import Path

from evaluate_evidence import evaluate_groups
from rag import ROOT, DenseRetriever, answer_packet, build_packet, missing_dates, sha256
from rag_generate import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--qid", action="append")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--output", type=Path, default=ROOT/"runs/rag_smoke.json")
    args = parser.parse_args()
    key = os.environ.get("OPENAI_API_KEY")
    if args.generate and (not key or not args.model):
        parser.error("--generate 실행에는 OPENAI_API_KEY와 OPENAI_MODEL이 필요합니다.")
    snapshot = ROOT/"eval/snapshots/v2-development-11q-2026-08-27"
    for name, expected in json.loads((snapshot/"checksums.json").read_text()).items():
        if sha256(snapshot/name) != expected:
            raise ValueError(f"고정 스냅샷이 변경됐습니다: {name}")
    questions = [json.loads(line) for line in (snapshot/"questions.v2.draft.jsonl").read_text().splitlines()]
    if args.qid:
        if set(args.qid) - {q["qid"] for q in questions}:
            parser.error("평가셋에 없는 qid입니다.")
        questions = [q for q in questions if q["qid"] in args.qid]
    baseline = {q["qid"]: q for q in json.loads((snapshot/"dense_rankings_v2_current.json").read_text())["per_question"]}
    retriever = DenseRetriever()
    results = []
    for question in questions:
        # Pass only visible question text into retrieval and generation.
        hits = retriever.search(question["question"], 5)
        packet = build_packet(question["question"], hits)
        if args.generate:
            result = answer_packet(packet, lambda p: generate(p, args.model, key))
        else:
            result = {"status": "retrieved_only", "answer": None, "generation_called": False,
                      "missing_date_conditions": missing_dates(packet)}
        row = {"qid": question["qid"], "result": result, "context": packet}
        if question["qid"] in baseline:
            row["baseline_top5_matches"] = [h["chunk_id"] for h in hits] == baseline[question["qid"]]["ranked_ids"][:5]
            if not row["baseline_top5_matches"]:
                raise ValueError(f"원문 기준선과 다른 검색 순위: {question['qid']}")
            metrics = evaluate_groups(
                [s["chunk_id"] for s in packet["sources"]], question["evidence_groups"], ks=(5,))
            row["context_evidence_metrics"] = {key: value for key, value in metrics.items() if key.endswith("@5")}
        results.append(row)
        print(question["qid"], result["status"], f"sources={len(packet['sources'])}")
    report = {"mode": "live_generation" if args.generate else "real_retrieval_only",
              "answer_quality_evaluated": False, "n_questions": len(results), "retriever": retriever.config,
              "questions_sha256": sha256(snapshot/"questions.v2.draft.jsonl"), "per_question": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"저장: {args.output}")
    return int(any(row["result"]["status"] in ("generation_error", "validation_failed") for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
