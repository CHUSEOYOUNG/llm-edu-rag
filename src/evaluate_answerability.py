"""Evaluate a Dense top-score baseline for answerability abstention.

This is a development diagnostic. It must not be treated as a held-out estimate
or as proof that a retrieved passage answers the question.
"""

import argparse
import hashlib
import json
from pathlib import Path

from rag import ROOT, DenseRetriever
from search_app import STRONG_CANDIDATE_THRESHOLD


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_questions(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if (not rows or len({row.get("qid") for row in rows}) != len(rows)
            or any(row.get("type") not in {"fact", "cond", "table", "unans"}
                   for row in rows)):
        raise ValueError("answerability 평가 질문을 확인하세요.")
    return [{"qid": row["qid"], "question": row["question"],
             "answerable": row["type"] != "unans"} for row in rows]


def metrics(rows, threshold):
    if not rows or {row["answerable"] for row in rows} != {False, True}:
        raise ValueError("answerable과 unanswerable 질문이 모두 필요합니다.")
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for row in rows:
        predicted = row["top_score"] >= threshold
        if predicted and row["answerable"]:
            counts["tp"] += 1
        elif predicted:
            counts["fp"] += 1
        elif row["answerable"]:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    recall = counts["tp"] / (counts["tp"] + counts["fn"])
    specificity = counts["tn"] / (counts["tn"] + counts["fp"])
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
    accuracy = (counts["tp"] + counts["tn"]) / len(rows)
    return {**counts, "precision": precision, "recall": recall,
            "specificity": specificity,
            "balanced_accuracy": (recall + specificity) / 2,
            "accuracy": accuracy}


def roc_auc(rows):
    positives = [row["top_score"] for row in rows if row["answerable"]]
    negatives = [row["top_score"] for row in rows if not row["answerable"]]
    if not positives or not negatives:
        raise ValueError("ROC-AUC에는 두 라벨이 모두 필요합니다.")
    wins = sum(1 if positive > negative else .5 if positive == negative else 0
               for positive in positives for negative in negatives)
    return wins / (len(positives) * len(negatives))


def best_threshold(rows):
    scores = sorted({row["top_score"] for row in rows})
    candidates = [scores[0] - 1e-9, *[(left + right) / 2
                    for left, right in zip(scores, scores[1:])], scores[-1] + 1e-9]
    evaluated = [(threshold, metrics(rows, threshold)) for threshold in candidates]
    return max(evaluated, key=lambda item: (
        item[1]["balanced_accuracy"], item[1]["accuracy"], item[0]))


def run(root=ROOT):
    questions_path = root / "eval/questions.v2.draft.jsonl"
    questions = read_questions(questions_path)
    retriever = DenseRetriever(root)
    rows = []
    for question in questions:
        hit = retriever.search(question["question"], 1)[0]
        rows.append({**question, "top_score": hit["score"],
                     "top_chunk_id": hit["chunk_id"]})
    threshold, best = best_threshold(rows)
    return {
        "experiment": "dense_top_score_answerability_baseline",
        "status": "development_diagnostic_not_held_out",
        "n_questions": len(rows),
        "n_answerable": sum(row["answerable"] for row in rows),
        "n_unanswerable": sum(not row["answerable"] for row in rows),
        "model": retriever.config["model"],
        "index_text": retriever.config["index_text"],
        "questions_sha256": sha256(questions_path),
        "roc_auc": roc_auc(rows),
        "service_review_threshold": STRONG_CANDIDATE_THRESHOLD,
        "service_threshold_metrics": metrics(rows, STRONG_CANDIDATE_THRESHOLD),
        "best_development_threshold": threshold,
        "best_development_metrics": best,
        "per_question": rows,
        "decision": "review_flag_only_do_not_auto_reject",
        "limitations": [
            "Only two unanswerable questions are labeled.",
            "The threshold and metrics use the same small development set.",
            "Dense similarity measures topical closeness, not whether a passage entails an answer.",
            "The service keeps low-score candidates visible and never labels them unanswerable.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = run(args.root)
    output = args.root / "experiments/answerability_baseline.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    current = result["service_threshold_metrics"]
    print(f"ROC-AUC: {result['roc_auc']:.3f}")
    print(f"서비스 검토 기준 {result['service_review_threshold']:.3f}: "
          f"recall={current['recall']:.3f}, specificity={current['specificity']:.3f}")
    print("결론: 자동 거절에는 사용하지 않고 검토 안내에만 사용합니다.")
    print(f"결과: {output}")


if __name__ == "__main__":
    main()
