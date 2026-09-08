"""Choose an answerability review threshold on development data and test it once."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluate_answerability import best_threshold, metrics, roc_auc
from rag import ROOT, DenseRetriever
from search_app import STRONG_CANDIDATE_THRESHOLD, assess_results


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_benchmark(path, chunk_ids):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    required = {"qid", "split", "answerable", "difficulty", "question", "evidence_chunk_ids"}
    if not rows or any(set(row) != required for row in rows):
        raise ValueError("answerability benchmark 필드를 확인하세요.")
    if len({row["qid"] for row in rows}) != len(rows):
        raise ValueError("중복 answerability qid가 있습니다.")
    for row in rows:
        if (row["split"] not in {"development", "test"}
                or type(row["answerable"]) is not bool
                or row["difficulty"] not in {"in_domain", "hard_negative", "out_of_domain"}
                or not isinstance(row["question"], str) or not row["question"].strip()
                or not isinstance(row["evidence_chunk_ids"], list)):
            raise ValueError("answerability benchmark 값을 확인하세요.")
        evidence = row["evidence_chunk_ids"]
        if row["answerable"] != bool(evidence) or not set(evidence) <= chunk_ids:
            raise ValueError(f"{row['qid']}의 답변 가능 라벨과 근거를 확인하세요.")
    for split in ("development", "test"):
        part = [row for row in rows if row["split"] == split]
        if {row["answerable"] for row in part} != {False, True}:
            raise ValueError(f"{split}에 두 라벨이 모두 필요합니다.")
    return rows


def evaluate_by_difficulty(rows, threshold):
    result = {}
    for difficulty in ("in_domain", "hard_negative", "out_of_domain"):
        subset = [row for row in rows if row["difficulty"] == difficulty]
        if not subset:
            continue
        predicted_strong = sum(row["top_score"] >= threshold for row in subset)
        result[difficulty] = {
            "n": len(subset),
            "strong_candidate": predicted_strong,
            "review_recommended": len(subset) - predicted_strong,
        }
    return result


def service_policy_metrics(rows):
    policy_rows = [{**row, "top_score": 1.0 if assess_results(
        row["question"], [{"score": row["top_score"]}])["level"] == "strong_candidate" else 0.0}
        for row in rows]
    return metrics(policy_rows, 0.5)


def run(root=ROOT):
    import numpy as np

    benchmark_path = root / "eval/answerability.v2.jsonl"
    chunks_path = root / "data/processed/chunks.jsonl"
    chunks = [json.loads(line) for line in chunks_path.read_text().splitlines() if line.strip()]
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    if len(by_id) != len(chunks):
        raise ValueError("중복 청크 ID가 있습니다.")
    rows = read_benchmark(benchmark_path, set(by_id))
    retriever = DenseRetriever(root)
    vectors = retriever.model.encode(
        [row["question"] for row in rows], normalize_embeddings=True, batch_size=8)
    similarities = vectors @ retriever.matrix.T
    scored = []
    for row, scores in zip(rows, similarities):
        index = int(np.argmax(scores))
        scored.append({**row, "top_score": float(scores[index]),
                       "top_chunk_id": chunks[index]["chunk_id"]})

    development = [row for row in scored if row["split"] == "development"]
    test = [row for row in scored if row["split"] == "test"]
    threshold, development_metrics = best_threshold(development)
    return {
        "experiment": "answerability_development_test_split",
        "status": "development_split_not_independent_external_test",
        "model": retriever.config["model"],
        "index_text": retriever.config["index_text"],
        "benchmark_sha256": sha256(benchmark_path),
        "chunks_sha256": sha256(chunks_path),
        "selection": "maximize development balanced accuracy; ties prefer accuracy then higher threshold",
        "selected_threshold": threshold,
        "development": {
            "n": len(development), "roc_auc": roc_auc(development),
            "metrics": development_metrics,
            "by_difficulty": evaluate_by_difficulty(development, threshold),
        },
        "test": {
            "n": len(test), "roc_auc": roc_auc(test),
            "metrics": metrics(test, threshold),
            "by_difficulty": evaluate_by_difficulty(test, threshold),
        },
        "current_service_threshold": {
            "value": STRONG_CANDIDATE_THRESHOLD,
            "test_metrics": metrics(test, STRONG_CANDIDATE_THRESHOLD),
        },
        "scope_aware_service_policy": {
            "signals": ["dense_top_score", "local_information_expression"],
            "test_metrics": service_policy_metrics(test),
        },
        "per_question": scored,
        "limitations": [
            "The benchmark was authored from the same six-document corpus and has no expert review.",
            "The test split was not used to select this threshold, but the retriever was developed before this split existed.",
            "Unanswerable labels mean the requested fact is absent from this fixed local corpus, not universally unknowable.",
            "Similarity is only a review signal and must not be presented as verified answerability.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = run(args.root)
    output = args.root / "experiments/answerability_split_v2.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"선택 기준: {result['selected_threshold']:.3f}")
    for split in ("development", "test"):
        part = result[split]
        print(f"{split}: AUC={part['roc_auc']:.3f}, "
              f"balanced_accuracy={part['metrics']['balanced_accuracy']:.3f}, "
              f"recall={part['metrics']['recall']:.3f}, "
              f"specificity={part['metrics']['specificity']:.3f}")
    print(f"결과: {output}")


if __name__ == "__main__":
    main()
