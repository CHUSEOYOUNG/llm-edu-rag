"""Evaluate automatic scope/content query separation on the frozen dev set."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from evaluate_evidence import DEPTH, ROOT, average, evaluate_groups, read_jsonl, sha256
from query_planning import interleave_rankings, supplemental_content_query

def run(root: Path = ROOT) -> dict:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    from sentence_transformers import SentenceTransformer

    config = json.loads((root / "config/dense_index.json").read_text())
    manifest = json.loads((root / "eval/questions.v2.draft.manifest.json").read_text())
    question_path = root / "eval/questions.v2.draft.jsonl"
    chunks_path = root / "data/processed/chunks.jsonl"
    embeddings_path = root / "data/processed/embeddings.npy"
    for path, key in ((chunks_path, "chunks_sha256"),
                      (embeddings_path, "embedding_sha256")):
        if sha256(path) != config[key]:
            raise ValueError(f"현재 Dense 색인 지문이 다릅니다: {path}")
    if config["index_text"] != "body":
        raise ValueError("body-only Dense 색인 실험만 지원합니다.")

    chunks = read_jsonl(chunks_path)
    questions = read_jsonl(question_path)
    targets = [q for q in questions if q["qid"] in manifest["reviewed_qids"]]
    if not targets or any(not q.get("evidence_groups") for q in targets):
        raise ValueError("검토된 평가 질문과 근거 그룹을 확인하세요.")
    if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
        raise ValueError("중복 청크 ID가 있습니다.")
    matrix = np.load(embeddings_path, allow_pickle=False)
    if matrix.ndim != 2 or len(matrix) != len(chunks) or not np.isfinite(matrix).all():
        raise ValueError("현재 Dense 임베딩을 확인하세요.")
    model = SentenceTransformer(config["model"], local_files_only=True)

    rows = []
    planned = [(q, supplemental_content_query(q["question"])) for q in targets]
    all_queries = [q["question"] for q in targets]
    all_queries.extend(query for _, query in planned if query)
    vectors = iter(model.encode(all_queries, normalize_embeddings=True))
    original_vectors = [next(vectors) for _ in targets]
    supplemental_vectors = iter(vectors)
    original_rankings = {}
    for question, vector in zip(targets, original_vectors):
        scores = matrix @ vector
        order = np.argsort(-scores)[:DEPTH]
        original_rankings[question["qid"]] = [chunks[index]["chunk_id"] for index in order]

    for question, content_query in planned:
        original_ids = original_rankings[question["qid"]]
        original_hits = [{"chunk_id": cid} for cid in original_ids]
        if content_query:
            scores = matrix @ next(supplemental_vectors)
            order = np.argsort(-scores)[:DEPTH]
            content_hits = [{"chunk_id": chunks[index]["chunk_id"]} for index in order]
            ranked = [hit["chunk_id"] for hit in interleave_rankings(
                [content_hits, original_hits], DEPTH
            )]
        else:
            ranked = original_ids
        rows.append({
            "qid": question["qid"],
            "original_question": question["question"],
            "supplemental_query": content_query,
            "metrics": evaluate_groups(ranked, question["evidence_groups"]),
            "ranked_ids": ranked,
        })

    baseline_rows = [{
        "qid": q["qid"],
        "metrics": evaluate_groups(original_rankings[q["qid"]], q["evidence_groups"]),
    } for q in targets]
    return {
        "experiment": "automatic_scope_content_query_separation",
        "status": "development_ablation_not_held_out",
        "corpus": "current validated Dense index",
        "n_questions": len(targets),
        "changed_qids": [row["qid"] for row in rows if row["supplemental_query"]],
        "fusion": "supplemental-first stable round-robin with original Dense ranking",
        "inputs": {
            "model": config["model"],
            "index_text": config["index_text"],
            "questions_sha256": sha256(question_path),
            "chunks_sha256": sha256(chunks_path),
            "embedding_sha256": sha256(embeddings_path),
        },
        "variants": {
            "dense_original": {"overall": average([r["metrics"] for r in baseline_rows]),
                               "per_question": baseline_rows},
            "scope_content_separated": {"overall": average([r["metrics"] for r in rows]),
                                        "per_question": rows},
        },
        "limitations": [
            "The same 11-question development set was used for prior diagnosis.",
            "Only scope-heavy content questions receive a supplemental query.",
            "This measures evidence retrieval, not temporal applicability or answer correctness.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.root)
    output = args.output or args.root / "experiments/ablation_scope_query.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for name, variant in result["variants"].items():
        print(name, json.dumps(variant["overall"], ensure_ascii=False))
    print("changed:", ", ".join(result["changed_qids"]) or "none")
    print("output:", output)


if __name__ == "__main__":
    main()
