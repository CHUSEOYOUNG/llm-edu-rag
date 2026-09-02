"""Compare NumPy Dense search with a persistent local Qdrant vector store."""

import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import statistics
import time

from evaluate_evidence import average, evaluate_groups, read_jsonl, validate_annotations
from qdrant_store import (COLLECTION, DEFAULT_CONFIG, DEFAULT_STORAGE,
                           QdrantVectorStore, build_index, load_dense_assets)

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ID = "v2-development-11q-2026-08-27"
TOP_K = 20
REPEATS = 30
KS = (1, 5, 10, 20)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def directory_size(path):
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def display_storage_path(storage, root):
    try:
        return str(Path(storage).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return Path(storage).name


def numpy_search(matrix, chunks, vector, k=TOP_K):
    import numpy as np

    scores = matrix @ vector
    order = np.argsort(-scores)[:k]
    return [{**chunks[index], "score": float(scores[index]), "point_id": int(index)}
            for index in order]


def latency_summary(values):
    if not values or not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("검색 지연시간 표본을 확인하세요.")
    ordered = sorted(values)
    p95_index = math.ceil(0.95 * len(ordered)) - 1
    return {
        "samples": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[p95_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def timed_search(search, vector):
    started = time.perf_counter()
    hits = search(vector)
    return hits, (time.perf_counter() - started) * 1000


def content_sequence(hits):
    return [(hit["path"], hit["body"]) for hit in hits]


def run(root=ROOT, storage=DEFAULT_STORAGE, config_path=DEFAULT_CONFIG,
        repeats=REPEATS):
    if repeats < 1:
        raise ValueError("반복 횟수는 1 이상이어야 합니다.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    from sentence_transformers import SentenceTransformer

    snapshot = root / "eval/snapshots" / SNAPSHOT_ID
    manifest = json.loads((snapshot / "questions.v2.draft.manifest.json").read_text())
    questions_path = snapshot / "questions.v2.draft.jsonl"
    rankings_path = snapshot / "dense_rankings_v2_current.json"
    if sha256(questions_path) != manifest["v2_sha256"]:
        raise ValueError("고정 평가 질문의 지문이 다릅니다.")
    dense_config, chunks, matrix = load_dense_assets(root)
    questions = read_jsonl(questions_path)
    validate_annotations(read_jsonl(root / "eval/questions.jsonl"), questions,
                         chunks, manifest)
    questions = [question for question in questions
                 if question["qid"] in manifest["reviewed_qids"]]
    frozen = json.loads(rankings_path.read_text())
    expected_metadata = {
        "model": manifest["model"],
        "index_text": "body",
        "chunks_sha256": manifest["chunks_sha256"],
        "questions_sha256": manifest["v2_sha256"],
        "embedding_sha256": manifest["embedding_sha256"],
        "evaluation_depth": TOP_K,
    }
    if any(frozen.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("고정 Dense 순위의 입력 지문이 다릅니다.")
    frozen_rows = {row["qid"]: row for row in frozen["per_question"]}
    if (len(frozen_rows) != len(frozen["per_question"])
            or set(frozen_rows) != {question["qid"] for question in questions}):
        raise ValueError("고정 Dense 순위의 질문 구성이 다릅니다.")

    model = SentenceTransformer(dense_config["model"], local_files_only=True)
    vectors = np.asarray(model.encode(
        [question["question"] for question in questions], normalize_embeddings=True))
    if (vectors.ndim != 2 or vectors.shape[1] != matrix.shape[1]
            or not np.isfinite(vectors).all()):
        raise ValueError("질문 임베딩을 확인하세요.")

    dense_hits = []
    for question, vector in zip(questions, vectors):
        hits = numpy_search(matrix, chunks, vector)
        ids = [hit["chunk_id"] for hit in hits]
        if ids != frozen_rows[question["qid"]]["ranked_ids"]:
            raise ValueError(f'Dense 기준선을 재현하지 못했습니다: {question["qid"]}')
        dense_hits.append(hits)

    build_started = time.perf_counter()
    index_config = build_index(root, storage, config_path, recreate=True)
    build_ms = (time.perf_counter() - build_started) * 1000
    disk_bytes = directory_size(storage)

    reload_started = time.perf_counter()
    store = QdrantVectorStore(root, storage, config_path)
    reload_open_ms = (time.perf_counter() - reload_started) * 1000
    try:
        _, first_query_ms = timed_search(
            lambda vector: store.search_vector(vector, TOP_K), vectors[0])
        qdrant_rows, dense_rows = [], []
        exact_top5 = exact_top20 = 0
        content_top5 = content_top20 = 0
        overlaps, score_differences = [], []
        for question, vector, baseline_hits in zip(questions, vectors, dense_hits):
            qdrant_hits = store.search_vector(vector, TOP_K)
            dense_ids = [hit["chunk_id"] for hit in baseline_hits]
            qdrant_ids = [hit["chunk_id"] for hit in qdrant_hits]
            exact_top5 += dense_ids[:5] == qdrant_ids[:5]
            exact_top20 += dense_ids == qdrant_ids
            content_top5 += (content_sequence(baseline_hits[:5])
                             == content_sequence(qdrant_hits[:5]))
            content_top20 += (content_sequence(baseline_hits)
                              == content_sequence(qdrant_hits))
            overlaps.append(len(set(dense_ids) & set(qdrant_ids)) / TOP_K)
            dense_score = {hit["chunk_id"]: hit["score"] for hit in baseline_hits}
            score_differences.extend(
                abs(dense_score[hit["chunk_id"]] - hit["score"])
                for hit in qdrant_hits if hit["chunk_id"] in dense_score
            )
            groups = question["evidence_groups"]
            dense_metrics = evaluate_groups(dense_ids, groups, ks=KS)
            qdrant_metrics = evaluate_groups(qdrant_ids, groups, ks=KS)
            dense_rows.append({
                "qid": question["qid"], "metrics": dense_metrics,
                "ranked_ids": dense_ids,
            })
            qdrant_rows.append({
                "qid": question["qid"], "metrics": qdrant_metrics,
                "ranked_ids": qdrant_ids,
                "scores": [hit["score"] for hit in qdrant_hits],
                "top20_overlap": overlaps[-1],
                "top5_sequence_equal": dense_ids[:5] == qdrant_ids[:5],
                "top20_sequence_equal": dense_ids == qdrant_ids,
            })

        numpy_latencies, qdrant_latencies = [], []
        numpy_fn = lambda vector: numpy_search(matrix, chunks, vector, TOP_K)
        qdrant_fn = lambda vector: store.search_vector(vector, TOP_K)
        numpy_fn(vectors[0])
        qdrant_fn(vectors[0])
        for repeat in range(repeats):
            for vector in vectors:
                if repeat % 2:
                    _, elapsed = timed_search(qdrant_fn, vector)
                    qdrant_latencies.append(elapsed)
                    _, elapsed = timed_search(numpy_fn, vector)
                    numpy_latencies.append(elapsed)
                else:
                    _, elapsed = timed_search(numpy_fn, vector)
                    numpy_latencies.append(elapsed)
                    _, elapsed = timed_search(qdrant_fn, vector)
                    qdrant_latencies.append(elapsed)
    finally:
        store.close()

    dense_overall = average([row["metrics"] for row in dense_rows])
    qdrant_overall = average([row["metrics"] for row in qdrant_rows])
    numpy_asset_bytes = sum((root / relative).stat().st_size for relative in (
        "data/processed/chunks.jsonl", "data/processed/embeddings.npy"))
    return {
        "experiment": "vector_store_ablation",
        "status": "development_infrastructure_comparison_not_held_out",
        "snapshot_id": SNAPSHOT_ID,
        "n_questions": len(questions),
        "n_evidence_groups": sum(len(question["evidence_groups"])
                                 for question in questions),
        "index": {
            **index_config,
            "qdrant_client_version": version("qdrant-client"),
            "mode_detail": "persistent local mode; exact brute-force search",
            "storage_path": display_storage_path(storage, root),
            "disk_bytes": disk_bytes,
            "numpy_chunks_and_embeddings_bytes": numpy_asset_bytes,
            "storage_size_ratio_vs_numpy_assets": disk_bytes / numpy_asset_bytes,
            "build_ms": build_ms,
            "reload_open_ms": reload_open_ms,
            "first_query_after_reload_ms": first_query_ms,
        },
        "inputs": {
            "questions_sha256": sha256(questions_path),
            "chunks_sha256": sha256(root / "data/processed/chunks.jsonl"),
            "embedding_sha256": sha256(root / "data/processed/embeddings.npy"),
            "dense_rankings_sha256": sha256(rankings_path),
        },
        "equivalence": {
            "exact_top5_sequences": exact_top5,
            "exact_top20_sequences": exact_top20,
            "content_equivalent_top5_sequences": content_top5,
            "content_equivalent_top20_sequences": content_top20,
            "questions": len(questions),
            "mean_top20_overlap": statistics.mean(overlaps),
            "min_top20_overlap": min(overlaps),
            "max_shared_score_abs_diff": max(score_differences, default=0.0),
            "retrieval_metrics_equal": dense_overall == qdrant_overall,
        },
        "benchmark": {
            "top_k": TOP_K,
            "repeats_per_question": repeats,
            "scope": "vector search and result materialization; query embedding excluded",
            "numpy": latency_summary(numpy_latencies),
            "qdrant_local": latency_summary(qdrant_latencies),
        },
        "variants": {
            "numpy": {"overall": dense_overall, "per_question": dense_rows},
            "qdrant_local": {"overall": qdrant_overall, "per_question": qdrant_rows},
        },
        "limitations": [
            "Qdrant local mode performs exact brute-force search, not server HNSW search.",
            "Latency is a single-machine microbenchmark on a 1,331-point collection.",
            "The same 11-question development set was used for earlier design work.",
            "This measures retrieval infrastructure, not generated-answer correctness.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--storage", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args()
    result = run(args.root, args.storage, args.config, args.repeats)
    output = args.root / "experiments/ablation_vector_store.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    equivalent = result["equivalence"]
    benchmark = result["benchmark"]
    print(f'Exact top-20: {equivalent["exact_top20_sequences"]}/'
          f'{equivalent["questions"]}')
    print(f'Mean top-20 overlap: {equivalent["mean_top20_overlap"]:.3f}')
    print(f'NumPy mean: {benchmark["numpy"]["mean_ms"]:.3f} ms')
    print(f'Qdrant local mean: {benchmark["qdrant_local"]["mean_ms"]:.3f} ms')
    print(f'Qdrant disk: {result["index"]["disk_bytes"] / 1024 / 1024:.2f} MiB')
    print(f"결과: {output}")


if __name__ == "__main__":
    main()
