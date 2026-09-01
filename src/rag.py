"""Minimal cited RAG CLI; original-query Dense retrieval remains the default.

Default runs perform real local retrieval without API calls; --generate opts in.
Citation validation checks literal provenance, not semantic entailment.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import unicodedata

from rag_generate import GenerationError, generate

ROOT = Path(__file__).resolve().parents[1]
DATE = r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일"
CONDITIONS = re.compile(
    DATE + r"|(?:국가교육위원회|교육부)\s*고시\s*제\d{4}-\d+호"
    r"|\d{4}\s*개정\s*교육과정|초등학교|중학교|고등학교"
    r"|\d+(?:[·~∼〜-]\d+)?학년|\d+년간"
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compact(text):
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text)).lower()


class DenseRetriever:
    def __init__(self, root=ROOT):
        import numpy as np
        self.config = json.loads((root / "config/dense_index.json").read_text())
        cp, ep = root / "data/processed/chunks.jsonl", root / "data/processed/embeddings.npy"
        if self.config["index_text"] != "body":
            raise ValueError("이 검색기는 body-only 색인을 사용합니다.")
        for path, key in ((cp, "chunks_sha256"), (ep, "embedding_sha256")):
            if sha256(path) != self.config[key]:
                raise ValueError(f"색인 지문 불일치: {path.name}. 코퍼스와 임베딩을 함께 점검하세요.")
        self.chunks = [json.loads(line) for line in cp.read_text().splitlines() if line.strip()]
        self.matrix = np.load(ep, allow_pickle=False)
        if (self.matrix.ndim != 2 or not self.chunks or len(self.matrix) != len(self.chunks)
                or not np.isfinite(self.matrix).all()
                or not np.allclose(np.linalg.norm(self.matrix, axis=1), 1, atol=1e-3)):
            raise ValueError("정규화된 유효 임베딩과 청크의 대응을 확인하세요.")
        if len({c["chunk_id"] for c in self.chunks}) != len(self.chunks):
            raise ValueError("중복 청크 ID가 있습니다.")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.config["model"], local_files_only=True)

    def search(self, question, k=5):
        import numpy as np
        if not question.strip() or not 1 <= k <= 20:
            raise ValueError("질문은 비어 있을 수 없고 top-k는 1~20이어야 합니다.")
        vector = self.model.encode([question], normalize_embeddings=True)[0]
        scores = self.matrix @ vector
        return [{**self.chunks[i], "score": float(scores[i])}
                for i in np.argsort(-scores)[:k]]


def build_packet(question, hits, max_context_chars=24000):
    if not isinstance(question, str) or not question.strip() or len(question) > 4000:
        raise ValueError("질문은 1~4000자로 입력하세요.")
    if max_context_chars < 1:
        raise ValueError("컨텍스트 크기는 양수여야 합니다.")
    sources, omitted, used, seen = [], [], 0, set()
    for rank, hit in enumerate(hits, 1):
        cid = hit["chunk_id"]
        if cid in seen:
            continue
        seen.add(cid)
        # Keep complete chunks: cutting a table or exception can change its meaning.
        size = sum(len(hit[field]) for field in ("body", "path", "doc_id"))
        if used + size > max_context_chars:
            omitted.append(cid)
            continue
        provenance = {field: hit[field] for field in ("page_start", "page_end") if field in hit}
        sources.append({"source_id": f"S{len(sources)+1}", "chunk_id": cid,
                        "retrieval_rank": rank, "score": hit["score"],
                        **{field: hit[field] for field in ("body", "path", "doc_id")},
                        **provenance})
        used += size
    return {
        "original_question": question, "search_query": question,
        "scope_conditions": list(dict.fromkeys(m.group() for m in CONDITIONS.finditer(question))),
        "scope_extraction": "partial_regex; original question remains authoritative",
        "scope_filters_applied": False, "sources": sources,
        "context_chars": used, "omitted_chunk_ids": omitted,
    }


def missing_dates(packet):
    # Conservative literal guard, not a temporal applicability proof.
    return [condition for condition in packet["scope_conditions"]
            if re.fullmatch(DATE, condition)
            and not any(compact(condition) in compact(source[field])
                        for source in packet["sources"] for field in ("body", "path"))]


def require_keys(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("생성 응답의 필드 구성이 올바르지 않습니다.")


def verify_evidence(items, by_id, require_body=False):
    if not isinstance(items, list) or not items:
        raise ValueError("인용 근거가 비어 있습니다.")
    checked = []
    for item in items:
        require_keys(item, ("source_id", "field", "quote"))
        sid, field, quote = item["source_id"], item["field"], item["quote"]
        if not all(isinstance(v, str) for v in (sid, field, quote)):
            raise ValueError("인용 값은 문자열이어야 합니다.")
        if sid not in by_id or field not in ("body", "path", "doc_id"):
            raise ValueError("검색 컨텍스트에 없는 출처입니다.")
        source = by_id[sid]
        if not quote.strip() or quote not in source[field]:
            raise ValueError("인용문이 제공된 출처 원문에 없습니다.")
        start = source[field].index(quote)
        checked.append({**item, "chunk_id": source["chunk_id"], "start": start, "end": start+len(quote)})
    if require_body and not any(item["field"] == "body" for item in checked):
        raise ValueError("사실 문장에는 본문 근거가 필요합니다.")
    return checked


def validate_answer(answer, packet):
    require_keys(answer, ("status", "claims", "scope_checks", "reason"))
    if (answer["status"] not in ("answered", "insufficient_evidence")
            or not isinstance(answer["claims"], list) or not isinstance(answer["scope_checks"], list)
            or not isinstance(answer["reason"], str)):
        raise ValueError("생성 응답의 상태 또는 자료형이 올바르지 않습니다.")
    if answer["status"] == "insufficient_evidence":
        if answer["claims"] or not answer["reason"].strip():
            raise ValueError("보류 응답에는 사실 답변을 포함할 수 없고 사유가 필요합니다.")
        return {"status": "insufficient_evidence", "answer": None,
                "reason": answer["reason"], "claims": [], "scope_checks": []}
    if not answer["claims"]:
        raise ValueError("답변 문장이 없습니다.")
    if missing_dates(packet):
        raise ValueError("질문의 날짜 조건을 확인할 자료가 없습니다.")
    by_id = {source["source_id"]: source for source in packet["sources"]}
    claims, scopes = [], []
    for claim in answer["claims"]:
        require_keys(claim, ("text", "evidence"))
        if (not isinstance(claim["text"], str) or not claim["text"].strip()
                or re.search(r"\[S\d+\]", claim["text"])):
            raise ValueError("문장이 비었거나 모델이 인용 표기를 직접 삽입했습니다.")
        claims.append({"text": claim["text"],
                       "evidence": verify_evidence(claim["evidence"], by_id, require_body=True)})
    for scope in answer["scope_checks"]:
        require_keys(scope, ("condition", "status", "evidence"))
        if not isinstance(scope["condition"], str) or scope["status"] != "supported":
            raise ValueError("적용 조건을 뒷받침하지 못한 답변입니다.")
        evidence = verify_evidence(scope["evidence"], by_id)
        if re.fullmatch(DATE, scope["condition"]) and not any(
                item["field"] in ("body", "path")
                and compact(scope["condition"]) in compact(item["quote"]) for item in evidence):
            raise ValueError("시행일 인용에 해당 날짜가 없습니다.")
        scopes.append({**scope, "evidence": evidence})
    conditions = [scope["condition"] for scope in scopes]
    if len(conditions) != len(set(conditions)) or set(conditions) != set(packet["scope_conditions"]):
        raise ValueError("질문의 적용 조건이 누락되거나 바뀌었습니다.")
    rendered = []
    for claim in claims:
        ids = list(dict.fromkeys(item["source_id"] for item in claim["evidence"]))
        rendered.append(claim["text"] + " " + " ".join(f"[{sid}]" for sid in ids))
    return {"status": "draft_answer", "answer": "\n".join(rendered), "reason": "",
            "claims": claims, "scope_checks": scopes,
            "validation": {"citation_ids_and_quotes": "passed",
                           "semantic_entailment": "not_verified",
                           "scope_applicability": "model_assessed_not_independently_verified"}}


def answer_packet(packet, generator):
    if not packet["sources"]:
        return {"status": "insufficient_evidence", "answer": None, "reason": "전달할 근거가 없습니다.",
                "generation_called": False}
    if missing_dates(packet):
        return {"status": "insufficient_evidence", "answer": None,
                "reason": "검색 컨텍스트에서 요청한 날짜를 확인하지 못했습니다.",
                "missing_date_conditions": missing_dates(packet), "generation_called": False}
    try:
        raw, provenance = generator(packet)
        result = validate_answer(raw, packet)
        return {**result, "generation_called": True, "generation": provenance}
    except ValueError as exc:
        return {"status": "validation_failed", "answer": None, "reason": str(exc), "generation_called": True}
    except GenerationError as exc:
        return {"status": "generation_error", "answer": None, "reason": str(exc), "generation_called": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=24000)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="검색만 실행(기본값, 호환 옵션)")
    mode.add_argument("--generate", action="store_true", help="유료 생성 API 호출을 명시적으로 허용")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    key = os.environ.get("OPENAI_API_KEY")
    if args.generate and (not key or not args.model):
        parser.error("--generate에는 OPENAI_API_KEY와 OPENAI_MODEL이 필요합니다. 검색만 하려면 --generate를 빼세요.")
    try:
        build_packet(args.question, [], args.max_context_chars)
        if not 1 <= args.top_k <= 20:
            raise ValueError("top-k는 1~20이어야 합니다.")
        retriever = DenseRetriever()
        packet = build_packet(args.question, retriever.search(args.question, args.top_k), args.max_context_chars)
        if not args.generate:
            result = {"status": "retrieved_only", "answer": None, "generation_called": False,
                      "missing_date_conditions": missing_dates(packet)}
        else:
            result = answer_packet(packet, lambda p: generate(p, args.model, key))
        result.update({"question": args.question, "context": packet, "retriever": retriever.config})
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(f"상태: {result['status']}")
        print(result.get("answer") or result.get("reason") or "검색만 완료했습니다. 생성된 답변은 없습니다.")
        for source in packet["sources"]:
            print(f"[{source['source_id']}] {source['doc_id']} — {source['path']}")
            if not args.generate:
                preview = source["body"][:700]
                print(preview + ("\n[본문 미리보기 생략: 전체는 --output JSON에 보존]" if len(source["body"]) > 700 else ""))
        if result["status"] == "draft_answer":
            print("주의: 인용 원문 일치만 검증했습니다. 답변 의미·적용 범위는 별도 검토가 필요합니다.")
        if result.get("missing_date_conditions"):
            print("날짜 근거 미확인: " + ", ".join(result["missing_date_conditions"]))
        return 1 if result["status"] in ("generation_error", "validation_failed") else 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"실행 오류: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
