#!/usr/bin/env python3
"""13문항 검색/생성 회귀 평가 실행기."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ollama_generate import (DEFAULT_MODEL, generate as generate_local,
                             require_runnable_model)  # noqa: E402
from rag_generate import GenerationError  # noqa: E402
from rag import DenseRetriever  # noqa: E402
from regression_eval import (print_scorecard, read_jsonl, regression_failures,
                             run_regression)  # noqa: E402
from search_app import SearchService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "eval/questions.v2.draft.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--generate", action="store_true", help="Ollama로 생성 평가도 실행")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/regression_current.json")
    parser.add_argument("--baseline", type=Path, help="감소 시 실패시킬 기존 평가 JSON")
    parser.add_argument("--tolerance", type=float, default=0.0)
    args = parser.parse_args()
    questions_path = args.questions.resolve()

    if args.baseline and not args.baseline.is_file():
        parser.error(f"기준 평가 파일을 찾을 수 없습니다: {args.baseline}")
    if args.generate:
        try:
            print(f"Ollama 모델 실구동 점검: {args.model}", flush=True)
            require_runnable_model(args.model)
        except GenerationError as exc:
            parser.error(str(exc))
    retriever = DenseRetriever(ROOT)
    generator = ((lambda packet: generate_local(packet, model=args.model))
                 if args.generate else None)
    service = SearchService(retriever, generator=generator,
                            generation_model=args.model if args.generate else None)
    result = run_regression(service, read_jsonl(questions_path),
                            top_k=args.top_k, generate=args.generate)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["configuration"] = {
        "questions": str(questions_path.relative_to(ROOT))
        if questions_path.is_relative_to(ROOT) else questions_path.name,
        "top_k": args.top_k,
        "retriever_model": retriever.config["model"],
        "index_text": retriever.config["index_text"],
        "generation_model": args.model if args.generate else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print_scorecard(result)
    print(f"\n결과: {args.output}")
    if not args.baseline:
        return 0
    failures = regression_failures(
        result, json.loads(args.baseline.read_text(encoding="utf-8")), args.tolerance
    )
    if failures:
        print("\n회귀 감지:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("\n기준 평가보다 낮아진 핵심 지표가 없습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
