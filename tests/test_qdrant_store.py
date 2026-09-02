import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ablation_vector_store import (content_sequence, display_storage_path,
                                     latency_summary, numpy_search)
from qdrant_store import QdrantVectorStore, build_index, sha256


def make_dense_fixture(root):
    processed = root / "data/processed"
    config_dir = root / "config"
    processed.mkdir(parents=True)
    config_dir.mkdir()
    chunks = [
        {"chunk_id": "a", "doc_id": "문서", "path": "첫째", "body": "가", "n_chars": 1},
        {"chunk_id": "b", "doc_id": "문서", "path": "둘째", "body": "나", "n_chars": 1},
        {"chunk_id": "c", "doc_id": "문서", "path": "셋째", "body": "다", "n_chars": 1,
         "page_start": 3, "page_end": 3},
    ]
    chunks_path = processed / "chunks.jsonl"
    chunks_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chunks))
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0], [2 ** -0.5, 2 ** -0.5]],
                        dtype=np.float32)
    embeddings_path = processed / "embeddings.npy"
    np.save(embeddings_path, matrix, allow_pickle=False)
    (config_dir / "dense_index.json").write_text(json.dumps({
        "model": "test-model",
        "index_text": "body",
        "chunks_sha256": sha256(chunks_path),
        "embedding_sha256": sha256(embeddings_path),
    }))
    return chunks, matrix


class QdrantStoreTests(unittest.TestCase):
    def test_index_is_persistent_and_returns_payload_after_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks, matrix = make_dense_fixture(root)
            storage = root / "qdrant"
            config = root / "config/qdrant_index.json"
            manifest = build_index(root, storage, config, recreate=True)
            self.assertEqual(manifest["point_count"], 3)
            self.assertEqual(manifest["vector_size"], 2)

            with QdrantVectorStore(root, storage, config) as store:
                hits = store.search_vector(np.asarray([1.0, 0.0], dtype=np.float32), 3)
            self.assertEqual([hit["chunk_id"] for hit in hits], ["a", "c", "b"])
            self.assertEqual(hits[0]["body"], "가")
            self.assertNotIn("vector", hits[0])

            dense = numpy_search(matrix, chunks, np.asarray([1.0, 0.0], dtype=np.float32), 3)
            self.assertEqual([hit["chunk_id"] for hit in dense], ["a", "c", "b"])

    def test_changed_dense_input_is_rejected_before_opening_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dense_fixture(root)
            storage = root / "qdrant"
            config = root / "config/qdrant_index.json"
            build_index(root, storage, config, recreate=True)
            with (root / "data/processed/chunks.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "지문 불일치"):
                QdrantVectorStore(root, storage, config)

    def test_benchmark_helpers_validate_samples_and_hide_external_paths(self):
        summary = latency_summary([1, 2, 3, 4, 5])
        self.assertEqual(summary["median_ms"], 3)
        self.assertEqual(summary["p95_ms"], 5)
        with self.assertRaises(ValueError):
            latency_summary([])
        self.assertEqual(display_storage_path(Path("/tmp/vector-db"), Path("/repo")),
                         "vector-db")
        duplicate_content = [
            {"chunk_id": "a", "path": "항목", "body": "같은 내용"},
            {"chunk_id": "b", "path": "항목", "body": "같은 내용"},
        ]
        self.assertEqual(content_sequence(duplicate_content),
                         [("항목", "같은 내용"), ("항목", "같은 내용")])


if __name__ == "__main__":
    unittest.main()
