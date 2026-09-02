"""Compare structure, structure+overlap, and fixed-window Dense retrieval."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

from chunk import chunk_section
from evaluate_evidence import DEPTH, average, read_jsonl, validate_annotations

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ID = "v2-development-11q-2026-08-27"
MODEL = "BAAI/bge-m3"
FIXED_SIZE = 800
OVERLAP_CHARS = 200
MIN_TAIL = 100


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def section_path(section):
    return " > ".join(filter(None, [
        section["path"],
        f'{section["number"]} {section["title"]}'.strip(),
    ]))


def structure_chunks(sections):
    chunks = []
    for index, section in enumerate(sections):
        chunks.extend(chunk_section(section, index))
    return [chunk for chunk in chunks if chunk["n_chars"] >= 30]


def overlap_chunks(sections, overlap=OVERLAP_CHARS):
    if overlap < 1:
        raise ValueError("overlap은 1자 이상이어야 합니다.")
    chunks = []
    for index, section in enumerate(sections):
        base = [chunk for chunk in chunk_section(section, index) if chunk["n_chars"] >= 30]
        previous = None
        for chunk in base:
            body = chunk["body"]
            pages = {page for key in ("page_start", "page_end")
                     if (page := chunk.get(key)) is not None}
            if previous is not None:
                prefix = previous["body"][-overlap:]
                first_space = prefix.find(" ")
                if first_space >= 0:
                    prefix = prefix[first_space + 1:]
                body = f"{prefix}\n\n{body}"
                pages.update(page for key in ("page_start", "page_end")
                             if (page := previous.get(key)) is not None)
            result = {
                **chunk,
                "chunk_id": f'{chunk["chunk_id"]}::overlap{overlap}',
                "body": body,
                "text": f'{chunk["path"]}\n\n{body}' if chunk["path"] else body,
                "n_chars": len(body),
            }
            if pages:
                result.update(page_start=min(pages), page_end=max(pages))
            chunks.append(result)
            previous = chunk
    return chunks


def fixed_window_chunks(sections, size=FIXED_SIZE, min_tail=MIN_TAIL):
    if size < 1 or not 0 <= min_tail < size:
        raise ValueError("fixed window 크기 설정을 확인하세요.")
    chunks = []
    for section_index, section in enumerate(sections):
        body = section["text"].strip()
        if not body:
            continue
        windows = [body[start:start + size] for start in range(0, len(body), size)]
        if len(windows) > 1 and len(windows[-1]) < min_tail:
            tail = windows.pop()
            windows[-1] += tail
        path = section_path(section)
        for part, window in enumerate(windows):
            chunks.append({
                "chunk_id": f'{section["doc_id"]}::s{section_index:03d}::{section["number"]}::fixed::{part}',
                "doc_id": section["doc_id"],
                "path": path,
                "text": f"{path}\n\n{window}" if path else window,
                "body": window,
                "n_chars": len(window),
                "part": part,
                "n_parts": len(windows),
            })
    return [chunk for chunk in chunks if chunk["n_chars"] >= 30]


def evidence_options(question, chunks):
    by_doc = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk["doc_id"]].append(chunk)
    groups = []
    for group in question["evidence_groups"]:
        matches = set()
        for evidence in group["alternatives"]:
            matches.update(
                chunk["chunk_id"] for chunk in by_doc[evidence["doc_id"]]
                if evidence["text"] in chunk["body"]
            )
        groups.append(matches)
    return groups


def evaluate_mapped_groups(ranked, groups, ks=(1, 5, 10), depth=DEPTH):
    if not groups or len(ranked) != len(set(ranked)) or depth < max(ks):
        raise ValueError("평가 입력을 확인하세요.")
    union = set().union(*groups)
    metrics = {}
    for k in ks:
        top = set(ranked[:k])
        covered = sum(bool(top & options) for options in groups)
        metrics[f"hit@{k}"] = float(bool(top & union))
        metrics[f"coverage@{k}"] = covered / len(groups)
        metrics[f"complete@{k}"] = float(covered == len(groups))
    metrics[f"mrr@{depth}"] = next(
        (1 / rank for rank, chunk_id in enumerate(ranked[:depth], 1)
         if chunk_id in union), 0.0
    )
    return metrics


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def load_or_encode(model, chunks, cache_dir, name, batch_size, rebuild=False):
    import numpy as np

    chunks_path = cache_dir / f"{name}.jsonl"
    matrix_path = cache_dir / f"{name}.npy"
    manifest_path = cache_dir / f"{name}.manifest.json"
    write_jsonl(chunks_path, chunks)
    chunk_hash = sha256(chunks_path)
    expected = {"model": MODEL, "chunks_sha256": chunk_hash, "index_text": "body"}
    if not rebuild and matrix_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if all(manifest.get(key) == value for key, value in expected.items()):
            matrix = np.load(matrix_path, allow_pickle=False)
            if matrix.ndim == 2 and len(matrix) == len(chunks) and np.isfinite(matrix).all():
                print(f"캐시 사용: {name} ({len(chunks):,}개)")
                return matrix, chunk_hash

    print(f"임베딩 생성: {name} ({len(chunks):,}개)")
    matrix = np.asarray(model.encode(
        [chunk["body"] for chunk in chunks], batch_size=batch_size,
        normalize_embeddings=True, show_progress_bar=True,
    ))
    if (matrix.ndim != 2 or len(matrix) != len(chunks)
            or not np.isfinite(matrix).all()
            or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-3)):
        raise ValueError(f"유효하지 않은 임베딩: {name}")
    np.save(matrix_path, matrix, allow_pickle=False)
    manifest_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n")
    return matrix, chunk_hash


def variant_summary(chunks, matrix, questions, question_vectors):
    import numpy as np

    ids = [chunk["chunk_id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("중복 청크 ID가 있습니다.")
    rows = []
    mapped_groups = total_groups = 0
    for question, vector in zip(questions, question_vectors):
        groups = evidence_options(question, chunks)
        mapped_groups += sum(bool(options) for options in groups)
        total_groups += len(groups)
        scores = matrix @ vector
        order = np.argsort(-scores)[:DEPTH]
        ranked = [ids[index] for index in order]
        rows.append({
            "qid": question["qid"],
            "mapped_groups": sum(bool(options) for options in groups),
            "total_groups": len(groups),
            "metrics": evaluate_mapped_groups(ranked, groups),
            "ranked_ids": ranked,
        })
    sizes = sorted(chunk["n_chars"] for chunk in chunks)
    return {
        "chunk_stats": {
            "count": len(chunks),
            "indexed_chars": sum(sizes),
            "min": sizes[0],
            "median": sizes[len(sizes) // 2],
            "max": sizes[-1],
        },
        "evidence_mapping": {"mapped_groups": mapped_groups, "total_groups": total_groups},
        "overall": average([row["metrics"] for row in rows]),
        "per_question": rows,
    }


def run(root=ROOT, batch_size=8, rebuild=False):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    from sentence_transformers import SentenceTransformer

    snapshot = root / "eval/snapshots" / SNAPSHOT_ID
    manifest = json.loads((snapshot / "questions.v2.draft.manifest.json").read_text())
    questions_path = snapshot / "questions.v2.draft.jsonl"
    if sha256(questions_path) != manifest["v2_sha256"]:
        raise ValueError("고정 평가 질문의 지문이 다릅니다.")
    current_chunks = read_jsonl(root / "data/processed/chunks.jsonl")
    questions = read_jsonl(questions_path)
    validate_annotations(read_jsonl(root / "eval/questions.jsonl"), questions,
                         current_chunks, manifest)
    questions = [question for question in questions
                 if question["qid"] in manifest["reviewed_qids"]]
    sections = read_jsonl(root / "data/processed/sections.jsonl")

    variants = {
        "structure": structure_chunks(sections),
        "structure_overlap_200": overlap_chunks(sections),
        "fixed_800": fixed_window_chunks(sections),
    }
    baseline = variants["structure"]
    for expected, actual in zip(current_chunks, baseline):
        for field in ("chunk_id", "doc_id", "path", "body"):
            if expected[field] != actual[field]:
                raise ValueError(f"현재 structure 기준선을 재현하지 못했습니다: {field}")
    if len(current_chunks) != len(baseline):
        raise ValueError("현재 structure 청크 수를 재현하지 못했습니다.")

    model = SentenceTransformer(MODEL, local_files_only=True)
    question_vectors = model.encode(
        [question["question"] for question in questions], normalize_embeddings=True)
    cache_dir = root / "data/processed/chunking_ablation"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, chunks in variants.items():
        if name == "structure" and not rebuild:
            matrix = np.load(root / "data/processed/embeddings.npy", allow_pickle=False)
            chunks_path = root / "data/processed/chunks.jsonl"
            chunk_hash = sha256(chunks_path)
            print(f"현재 기준선 사용: structure ({len(chunks):,}개)")
        else:
            matrix, chunk_hash = load_or_encode(
                model, chunks, cache_dir, name, batch_size, rebuild)
        result = variant_summary(chunks, matrix, questions, question_vectors)
        result["chunks_sha256"] = chunk_hash
        results[name] = result

    expected_rankings = {
        row["qid"]: row["ranked_ids"] for row in json.loads(
            (snapshot / "dense_rankings_v2_current.json").read_text())["per_question"]
    }
    for row in results["structure"]["per_question"]:
        if row["ranked_ids"] != expected_rankings[row["qid"]]:
            raise ValueError(f'고정 Dense 순위를 재현하지 못했습니다: {row["qid"]}')

    return {
        "experiment": "chunking_ablation",
        "status": "development_ablation_not_held_out",
        "snapshot_id": SNAPSHOT_ID,
        "n_questions": len(questions),
        "n_evidence_groups": sum(len(question["evidence_groups"])
                                 for question in questions),
        "model": MODEL,
        "index_text": "body",
        "questions_sha256": sha256(questions_path),
        "sections_sha256": sha256(root / "data/processed/sections.jsonl"),
        "baseline_rankings_reproduced": True,
        "settings": {
            "structure": "paragraph/table-aware; target 800, hard limit 1500 except tables",
            "structure_overlap_200": "structure chunks with up to 200 preceding characters",
            "fixed_800": "800-character windows within sections; no overlap; tail under 100 merged",
        },
        "limitations": [
            "The same 11-question development set was used for prior design work.",
            "Evidence is remapped by exact annotated text within the same document.",
            "An evidence span split across chunks is intentionally counted as unmappable.",
            "This evaluates retrieval, not generated-answer quality or latency at production scale.",
        ],
        "variants": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    result = run(args.root, args.batch_size, args.rebuild)
    output = args.root / "experiments/ablation_chunking.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("\nvariant                  chunks  mapped  Complete@5  Coverage@5  MRR@20")
    for name, variant in result["variants"].items():
        stats, mapping, metrics = (
            variant["chunk_stats"], variant["evidence_mapping"], variant["overall"])
        print(f'{name:24}{stats["count"]:7d}  '
              f'{mapping["mapped_groups"]:2d}/{mapping["total_groups"]:<2d}    '
              f'{metrics["complete@5"]:10.3f}  {metrics["coverage@5"]:10.3f}  '
              f'{metrics["mrr@20"]:6.3f}')
    print(f"결과: {output}")


if __name__ == "__main__":
    main()
