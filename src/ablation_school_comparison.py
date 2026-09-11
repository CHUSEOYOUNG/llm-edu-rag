"""Evaluate school-specific query decomposition on the current Dense index."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from evaluate_evidence import DEPTH, ROOT, average, evaluate_groups, read_jsonl, sha256
from query_planning import interleave_rankings, school_comparison_queries
from search_app import matches_school_level


def run(root: Path = ROOT) -> dict:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    from sentence_transformers import SentenceTransformer

    config = json.loads((root / "config/dense_index.json").read_text())
    manifest = json.loads((root / "eval/questions.v2.draft.manifest.json").read_text())
    questions_path = root / "eval/questions.v2.draft.jsonl"
    chunks_path = root / "data/processed/chunks.jsonl"
    embeddings_path = root / "data/processed/embeddings.npy"
    for path, key in ((chunks_path, "chunks_sha256"),
                      (embeddings_path, "embedding_sha256")):
        if sha256(path) != config[key]:
            raise ValueError(f"현재 Dense 색인 지문이 다릅니다: {path}")
    if config["index_text"] != "body":
        raise ValueError("body-only Dense 색인 실험만 지원합니다.")

    chunks = read_jsonl(chunks_path)
    questions = read_jsonl(questions_path)
    targets = [q for q in questions if q["qid"] in manifest["reviewed_qids"]]
    matrix = np.load(embeddings_path, allow_pickle=False)
    model = SentenceTransformer(config["model"], local_files_only=True)
    if matrix.ndim != 2 or len(matrix) != len(chunks):
        raise ValueError("현재 Dense 임베딩을 확인하세요.")

    plans = {q["qid"]: school_comparison_queries(q["question"]) for q in targets}
    encoded_queries = []
    for question in targets:
        encoded_queries.append(question["question"])
        encoded_queries.extend(row["query"] for row in plans[question["qid"]])
    vectors = iter(model.encode(encoded_queries, normalize_embeddings=True))
    baseline_rows, decomposed_rows = [], []
    for question in targets:
        original_scores = matrix @ next(vectors)
        original_order = np.argsort(-original_scores)[:DEPTH]
        original_hits = [{"chunk_id": chunks[index]["chunk_id"]}
                         for index in original_order]
        groups = question["evidence_groups"]
        baseline_rows.append({"qid": question["qid"],
                              "metrics": evaluate_groups(
                                  [hit["chunk_id"] for hit in original_hits], groups)})

        school_rankings = []
        for plan in plans[question["qid"]]:
            scores = matrix @ next(vectors)
            order = np.argsort(-scores)
            ranking = [{"chunk_id": chunks[index]["chunk_id"]} for index in order
                       if matches_school_level(chunks[index], plan["school_level"])][:DEPTH]
            school_rankings.append(ranking)
        if school_rankings:
            ranked = [hit["chunk_id"] for hit in interleave_rankings(
                [*school_rankings, original_hits], DEPTH
            )]
        else:
            ranked = [hit["chunk_id"] for hit in original_hits]
        decomposed_rows.append({
            "qid": question["qid"],
            "subqueries": plans[question["qid"]],
            "metrics": evaluate_groups(ranked, groups),
            "ranked_ids": ranked,
        })

    return {
        "experiment": "school_comparison_query_decomposition",
        "status": "development_ablation_not_held_out",
        "n_questions": len(targets),
        "changed_qids": [qid for qid, plan in plans.items() if plan],
        "fusion": "school-filtered subqueries first, then original; stable round-robin",
        "inputs": {
            "model": config["model"], "index_text": config["index_text"],
            "questions_sha256": sha256(questions_path),
            "chunks_sha256": sha256(chunks_path),
            "embedding_sha256": sha256(embeddings_path),
        },
        "variants": {
            "dense_original": {"overall": average([r["metrics"] for r in baseline_rows]),
                               "per_question": baseline_rows},
            "school_decomposed": {"overall": average([r["metrics"] for r in decomposed_rows]),
                                  "per_question": decomposed_rows},
        },
        "limitations": [
            "The same development questions were used to diagnose this failure class.",
            "School filters use existing document/path/body metadata heuristics.",
            "This evaluates evidence retrieval, not generated-answer correctness.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = run(args.root)
    output = args.root / "experiments/ablation_school_comparison.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for name, variant in result["variants"].items():
        print(name, json.dumps(variant["overall"], ensure_ascii=False))
    print("changed:", ", ".join(result["changed_qids"]) or "none")
    print("output:", output)


if __name__ == "__main__":
    main()
