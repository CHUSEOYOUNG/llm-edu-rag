"""Rerank the frozen Dense top-20 with a multilingual cross-encoder."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

from evaluate_evidence import average, evaluate_groups, read_jsonl, validate_annotations

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ID = "v2-development-11q-2026-08-27"
MODEL = "BAAI/bge-reranker-v2-m3"
MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
CANDIDATE_DEPTH = 20
MAX_LENGTH = 1024
BATCH_SIZE = 4
KS = (1, 5, 10, 20)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stable_rerank(candidate_ids, scores):
    if (not candidate_ids or len(candidate_ids) != len(scores)
            or len(candidate_ids) != len(set(candidate_ids))):
        raise ValueError("reranker 후보와 점수를 확인하세요.")
    numeric = [float(score) for score in scores]
    if not all(math.isfinite(score) for score in numeric):
        raise ValueError("reranker 점수는 유한한 숫자여야 합니다.")
    order = sorted(range(len(candidate_ids)), key=lambda index: (-numeric[index], index))
    return [candidate_ids[index] for index in order]


def first_evidence_ranks(ranked, groups):
    ranks = {}
    for group in groups:
        alternatives = {item["chunk_id"] for item in group["alternatives"]}
        ranks[group["group_id"]] = next(
            (rank for rank, chunk_id in enumerate(ranked, 1)
             if chunk_id in alternatives), None
        )
    return ranks


def synchronize(device, torch):
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize(device)


def resolve_device(requested, torch):
    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS를 사용할 수 없습니다.")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA를 사용할 수 없습니다.")
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pair_token_lengths(tokenizer, question, bodies):
    return [len(tokenizer(question, body, add_special_tokens=True,
                          truncation=False)["input_ids"])
            for body in bodies]


def score_pairs(model, tokenizer, question, bodies, device, torch,
                batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    scores = []
    synchronize(device, torch)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(bodies), batch_size):
            pairs = [(question, body) for body in bodies[start:start + batch_size]]
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation="only_second",
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs, return_dict=True).logits.reshape(-1)
            scores.extend(float(value) for value in logits.float().cpu())
    synchronize(device, torch)
    return scores, (time.perf_counter() - started) * 1000


def latency_summary(values):
    ordered = sorted(values)
    p95_index = math.ceil(0.95 * len(ordered)) - 1
    return {
        "total_ms": sum(values),
        "mean_ms_per_question": statistics.mean(values),
        "median_ms_per_question": statistics.median(values),
        "p95_ms_per_question": ordered[p95_index],
    }


def run(root=ROOT, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, device="auto"):
    if batch_size < 1 or max_length < 8:
        raise ValueError("batch size와 max length를 확인하세요.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    snapshot = root / "eval/snapshots" / SNAPSHOT_ID
    manifest = json.loads((snapshot / "questions.v2.draft.manifest.json").read_text())
    questions_path = snapshot / "questions.v2.draft.jsonl"
    chunks_path = root / "data/processed/chunks.jsonl"
    rankings_path = snapshot / "dense_rankings_v2_current.json"
    if sha256(questions_path) != manifest["v2_sha256"]:
        raise ValueError("고정 평가 질문의 지문이 다릅니다.")
    chunks = read_jsonl(chunks_path)
    questions = read_jsonl(questions_path)
    validate_annotations(read_jsonl(root / "eval/questions.jsonl"), questions,
                         chunks, manifest)
    questions = [question for question in questions
                 if question["qid"] in manifest["reviewed_qids"]]
    by_chunk = {chunk["chunk_id"]: chunk for chunk in chunks}
    if len(by_chunk) != len(chunks):
        raise ValueError("중복 청크 ID가 있습니다.")

    ranking_data = json.loads(rankings_path.read_text())
    expected_ranking_metadata = {
        "model": manifest["model"],
        "index_text": "body",
        "chunks_sha256": manifest["chunks_sha256"],
        "questions_sha256": manifest["v2_sha256"],
        "embedding_sha256": manifest["embedding_sha256"],
        "evaluation_depth": CANDIDATE_DEPTH,
    }
    if any(ranking_data.get(key) != value
           for key, value in expected_ranking_metadata.items()):
        raise ValueError("고정 Dense 후보의 입력 지문이 다릅니다.")
    ranking_rows = {row["qid"]: row for row in ranking_data["per_question"]}
    if (set(ranking_rows) != {question["qid"] for question in questions}
            or len(ranking_rows) != len(ranking_data["per_question"])):
        raise ValueError("고정 Dense 후보의 질문 구성이 다릅니다.")
    for row in ranking_rows.values():
        ids = row["ranked_ids"]
        if (len(ids) != CANDIDATE_DEPTH or len(ids) != len(set(ids))
                or not set(ids) <= set(by_chunk)):
            raise ValueError("Dense top-20 후보를 확인하세요.")

    model_path = Path(snapshot_download(
        repo_id=MODEL,
        revision=MODEL_REVISION,
        allow_patterns=list(MODEL_FILES),
        local_files_only=True,
    ))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True)
    selected_device = resolve_device(device, torch)
    model.to(selected_device)
    model.eval()

    first_question = questions[0]
    first_id = ranking_rows[first_question["qid"]]["ranked_ids"][0]
    score_pairs(model, tokenizer, first_question["question"],
                [by_chunk[first_id]["body"]], selected_device, torch,
                batch_size=1, max_length=max_length)

    dense_rows, reranked_rows, latencies, token_lengths = [], [], [], []
    for question in questions:
        candidate_ids = ranking_rows[question["qid"]]["ranked_ids"]
        bodies = [by_chunk[chunk_id]["body"] for chunk_id in candidate_ids]
        lengths = pair_token_lengths(tokenizer, question["question"], bodies)
        token_lengths.extend(lengths)
        scores, elapsed_ms = score_pairs(
            model, tokenizer, question["question"], bodies,
            selected_device, torch, batch_size, max_length)
        ranked = stable_rerank(candidate_ids, scores)
        groups = question["evidence_groups"]
        dense_rows.append({
            "qid": question["qid"],
            "metrics": evaluate_groups(candidate_ids, groups, ks=KS),
            "first_evidence_rank": first_evidence_ranks(candidate_ids, groups),
            "ranked_ids": candidate_ids,
        })
        score_by_id = dict(zip(candidate_ids, scores))
        reranked_rows.append({
            "qid": question["qid"],
            "metrics": evaluate_groups(ranked, groups, ks=KS),
            "first_evidence_rank": first_evidence_ranks(ranked, groups),
            "ranked_ids": ranked,
            "scores": [score_by_id[chunk_id] for chunk_id in ranked],
            "latency_ms": elapsed_ms,
        })
        latencies.append(elapsed_ms)
        print(f'{question["qid"]}: {elapsed_ms:,.0f} ms')

    return {
        "experiment": "cross_encoder_reranker",
        "status": "development_ablation_not_held_out",
        "snapshot_id": SNAPSHOT_ID,
        "n_questions": len(questions),
        "n_evidence_groups": sum(len(question["evidence_groups"])
                                 for question in questions),
        "candidate_retriever": "frozen BAAI/bge-m3 body-only Dense top-20",
        "candidate_depth": CANDIDATE_DEPTH,
        "reranker": {
            "model": MODEL,
            "revision": MODEL_REVISION,
            "model_sha256": sha256(model_path / "model.safetensors"),
            "input": "original question + chunk body",
            "max_length": max_length,
            "batch_size": batch_size,
            "dtype": str(next(model.parameters()).dtype),
            "device": selected_device,
            "pairs": len(token_lengths),
            "pairs_truncated": sum(length > max_length for length in token_lengths),
            "max_pair_tokens_before_truncation": max(token_lengths),
        },
        "inputs": {
            "questions_sha256": sha256(questions_path),
            "chunks_sha256": sha256(chunks_path),
            "dense_rankings_sha256": sha256(rankings_path),
        },
        "variants": {
            "dense": {
                "overall": average([row["metrics"] for row in dense_rows]),
                "per_question": dense_rows,
            },
            "dense_then_reranker": {
                "overall": average([row["metrics"] for row in reranked_rows]),
                "latency": latency_summary(latencies),
                "per_question": reranked_rows,
            },
        },
        "limitations": [
            "The same 11-question development set was used for earlier design work.",
            "The reranker can only reorder documents already present in Dense top-20.",
            "Latency is a single local-device run and excludes model loading and Dense retrieval.",
            "This evaluates annotated evidence retrieval, not generated-answer correctness.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = run(args.root, args.batch_size, args.max_length, args.device)
    output = args.root / "experiments/ablation_reranker.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("\nvariant              Complete@1  Complete@5  Complete@10  MRR@20")
    for name, variant in result["variants"].items():
        metrics = variant["overall"]
        print(f'{name:22}{metrics["complete@1"]:10.3f}  '
              f'{metrics["complete@5"]:10.3f}  {metrics["complete@10"]:11.3f}  '
              f'{metrics["mrr@20"]:6.3f}')
    latency = result["variants"]["dense_then_reranker"]["latency"]
    print(f'평균 rerank 지연: {latency["mean_ms_per_question"]:,.0f} ms/question')
    print(f"결과: {output}")


if __name__ == "__main__":
    main()
