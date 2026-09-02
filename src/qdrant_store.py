"""Build and query a persistent local Qdrant index from the frozen Dense vectors."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = "education_chunks"
DEFAULT_STORAGE = ROOT / "data/processed/qdrant"
DEFAULT_CONFIG = ROOT / "config/qdrant_index.json"
PAYLOAD_FIELDS = ("chunk_id", "doc_id", "path", "body", "n_chars",
                  "page_start", "page_end")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_dense_assets(root=ROOT):
    import numpy as np

    config = json.loads((root / "config/dense_index.json").read_text())
    chunks_path = root / "data/processed/chunks.jsonl"
    embeddings_path = root / "data/processed/embeddings.npy"
    if config.get("index_text") != "body":
        raise ValueError("Qdrant 색인은 body-only Dense 벡터를 사용합니다.")
    for path, key in ((chunks_path, "chunks_sha256"),
                      (embeddings_path, "embedding_sha256")):
        if sha256(path) != config.get(key):
            raise ValueError(f"Dense 입력 지문 불일치: {path.name}")
    chunks = [json.loads(line) for line in chunks_path.read_text().splitlines()
              if line.strip()]
    matrix = np.load(embeddings_path, allow_pickle=False)
    if (matrix.ndim != 2 or not chunks or len(chunks) != len(matrix)
            or not np.isfinite(matrix).all()
            or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-3)):
        raise ValueError("Dense 청크와 정규화 임베딩을 확인하세요.")
    if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
        raise ValueError("중복 청크 ID가 있습니다.")
    return config, chunks, matrix


def chunk_payload(chunk):
    return {field: chunk[field] for field in PAYLOAD_FIELDS if field in chunk}


def index_manifest(dense_config, matrix, point_count):
    return {
        "backend": "qdrant-local-persistent",
        "collection": COLLECTION,
        "distance": "Cosine",
        "vector_size": int(matrix.shape[1]),
        "point_count": point_count,
        "payload_fields": list(PAYLOAD_FIELDS),
        "source_model": dense_config["model"],
        "source_index_text": dense_config["index_text"],
        "chunks_sha256": dense_config["chunks_sha256"],
        "embedding_sha256": dense_config["embedding_sha256"],
        "usage": "local development and reproducibility; use Qdrant server for deployment",
    }


def build_index(root=ROOT, storage=DEFAULT_STORAGE, config_path=DEFAULT_CONFIG,
                recreate=False):
    from qdrant_client import QdrantClient, models

    dense_config, chunks, matrix = load_dense_assets(root)
    storage = Path(storage)
    if recreate and storage.exists():
        shutil.rmtree(storage)
    client = QdrantClient(path=str(storage))
    try:
        if client.collection_exists(COLLECTION):
            if not recreate:
                raise ValueError("Qdrant collection이 이미 있습니다. --recreate를 사용하세요.")
            client.delete_collection(COLLECTION)
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=int(matrix.shape[1]), distance=models.Distance.COSINE),
        )
        client.upload_collection(
            collection_name=COLLECTION,
            vectors=matrix,
            ids=list(range(len(chunks))),
            payload=[chunk_payload(chunk) for chunk in chunks],
            batch_size=128,
        )
        count = client.count(COLLECTION, exact=True).count
        if count != len(chunks):
            raise ValueError(f"Qdrant point 수 불일치: {count} != {len(chunks)}")
    finally:
        client.close()
    manifest = index_manifest(dense_config, matrix, len(chunks))
    if config_path is not None:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


class QdrantVectorStore:
    def __init__(self, root=ROOT, storage=DEFAULT_STORAGE, config_path=DEFAULT_CONFIG):
        from qdrant_client import QdrantClient

        self.config = json.loads(Path(config_path).read_text())
        dense_config, chunks, matrix = load_dense_assets(root)
        expected = index_manifest(dense_config, matrix, len(chunks))
        if self.config != expected:
            raise ValueError("Qdrant 설정과 현재 Dense 색인이 다릅니다. 색인을 다시 만드세요.")
        self.client = QdrantClient(path=str(storage))
        if (not self.client.collection_exists(COLLECTION)
                or self.client.count(COLLECTION, exact=True).count != len(chunks)):
            self.client.close()
            raise ValueError("Qdrant collection을 확인하세요.")

    def search_vector(self, vector, k=5):
        if not 1 <= k <= self.config["point_count"]:
            raise ValueError("검색 개수를 확인하세요.")
        response = self.client.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )
        hits = []
        for point in response.points:
            payload = point.payload or {}
            if set(("chunk_id", "doc_id", "path", "body")) - set(payload):
                raise ValueError("Qdrant payload에 검색 결과 필드가 없습니다.")
            hits.append({**payload, "score": float(point.score), "point_id": point.id})
        if len(hits) != k:
            raise ValueError("요청한 수만큼 Qdrant 결과를 받지 못했습니다.")
        return hits

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--storage", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()
    manifest = build_index(args.root, args.storage, args.config, args.recreate)
    print(f'Qdrant collection: {manifest["collection"]}')
    print(f'points: {manifest["point_count"]:,}, dimensions: {manifest["vector_size"]:,}')
    print(f"storage: {args.storage}")


if __name__ == "__main__":
    main()
