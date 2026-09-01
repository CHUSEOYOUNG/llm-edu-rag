"""Loopback-only evidence browser. No generation routes, keys, or paid API calls.

Run: python src/search_app.py, then open http://127.0.0.1:8765.
This standard-library development server is not intended for public deployment.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import quote, unquote, urlsplit
import re

from rag import ROOT, DenseRetriever, build_packet, compact, missing_dates

WEB = ROOT / "web"
MAX_REQUEST_BYTES = 16384
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.css": ("app.css", "text/css; charset=utf-8"),
          "/presentation.js": ("presentation.js", "text/javascript; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8")}
SCHOOL_LEVELS = {
    "all": None,
    "elementary": ("(초)", "초등학교"),
    "middle": ("(중)", "중학교"),
    "high": ("(고)", "고등학교"),
}


class BusyError(RuntimeError):
    pass


def keyword_terms(question):
    """Return 1-3 visible terms only for short keyword-style searches."""
    stripped = question.strip()
    if len(stripped) > 30 or not re.fullmatch(r"[0-9A-Za-z가-힣·~∼〜\-\s]+", stripped):
        return []
    terms = [compact(term) for term in re.findall(r"[0-9A-Za-z가-힣]+", stripped)]
    return terms if 1 <= len(terms) <= 3 and all(len(term) >= 2 for term in terms) else []


def keyword_rerank(question, hits):
    """Prefer literal section/grade matches for short queries; keep Dense ties stable."""
    terms = keyword_terms(question)
    if not terms:
        return hits

    ranked = []
    for dense_rank, hit in enumerate(hits):
        path = compact(hit["path"])
        body = compact(hit["body"])
        doc = compact(hit["doc_id"])
        if "(초)" in hit["doc_id"]:
            doc += "초등학교"
        if "(중)" in hit["doc_id"]:
            doc += "중학교"
        if "(고)" in hit["doc_id"]:
            doc += "고등학교"
        path_hits = sum(term in path for term in terms)
        doc_hits = sum(term in doc for term in terms)
        body_hits = sum(term in body for term in terms)
        all_terms = all(term in path or term in doc or term in body for term in terms)
        ranked.append((hit, all_terms, path_hits, doc_hits, body_hits, dense_rank))

    if not any(item[1] for item in ranked):
        return hits
    ranked.sort(key=lambda item: (
        not item[1], -item[2], -item[3],
        item[0]["path"].count(">") if item[2] else 999,
        len(item[0]["path"]) if item[2] else 999,
        -item[4], item[5],
    ))
    return [item[0] for item in ranked]


def matches_school_level(hit, school_level):
    """Match a grade-specific document or an explicit grade mention in a general one."""
    markers = SCHOOL_LEVELS[school_level]
    if markers is None:
        return True
    document_marker, visible_name = markers
    specific_markers = tuple(level[0] for level in SCHOOL_LEVELS.values() if level)
    if any(marker in hit["doc_id"] for marker in specific_markers):
        return document_marker in hit["doc_id"]
    return compact(visible_name) in compact(
        " ".join((hit["doc_id"], hit["path"], hit["body"]))
    )


def condition_audit(packet):
    """Literal occurrences only, never semantic applicability judgments."""
    return [{"condition": condition, "semantic_verdict": "not_verified", "sources": [
        {"source_id": source["source_id"],
         "fields": [field for field in ("body", "path", "doc_id")
                    if compact(condition) in compact(source[field])]}
        for source in packet["sources"]
    ]} for condition in packet["scope_conditions"]]


class SearchService:
    def __init__(self, retriever, source_root=ROOT / "data/raw"):
        self.retriever = retriever
        self.lock = threading.Lock()
        source_root = source_root.resolve()
        self.source_files = {}
        for doc_id in {chunk["doc_id"] for chunk in retriever.chunks}:
            candidate = (source_root / f"{doc_id}.pdf").resolve()
            if candidate.parent == source_root and candidate.is_file():
                self.source_files[doc_id] = candidate

    def source_path(self, doc_id):
        return self.source_files.get(doc_id)

    def add_source_links(self, packet):
        for source in packet["sources"]:
            page = source.get("page_start")
            if self.source_path(source["doc_id"]) and type(page) is int and page >= 1:
                source["source_url"] = f"/source/{quote(source['doc_id'], safe='')}.pdf#page={page}"

    def info(self):
        return {"mode": "local_retrieval_only", "generation_enabled": False,
                "model": self.retriever.config["model"], "index_text": "body",
                "short_keyword_rerank": True,
                "chunk_count": len(self.retriever.chunks),
                "document_count": len({c["doc_id"] for c in self.retriever.chunks})}

    def search(self, payload):
        if not isinstance(payload, dict) or set(payload) - {"question", "top_k", "school_level"}:
            raise ValueError("입력한 검색 내용을 확인해 주세요.")
        question, k = payload.get("question"), payload.get("top_k", 5)
        school_level = payload.get("school_level", "all")
        build_packet(question, [])
        if type(k) is not int or not 1 <= k <= 20:
            raise ValueError("검색 개수는 1~20 사이의 정수여야 합니다.")
        if school_level not in SCHOOL_LEVELS:
            raise ValueError("학교급 선택을 확인해 주세요.")
        if not self.lock.acquire(blocking=False):
            raise BusyError("다른 내용을 찾고 있어요. 잠시 후 다시 시도해 주세요.")
        started = time.perf_counter()
        try:
            terms = keyword_terms(question)
            if school_level != "all":
                candidate_k = len(self.retriever.chunks)
            else:
                candidate_k = min(len(self.retriever.chunks), max(k, 100)) if terms else k
            candidates = self.retriever.search(question, candidate_k)
            candidates = [hit for hit in candidates if matches_school_level(hit, school_level)]
            hits = keyword_rerank(question, candidates)[:k]
            packet = build_packet(question, hits)
            packet["scope_filters_applied"] = school_level != "all"
            packet["school_level_filter"] = school_level
            self.add_source_links(packet)
        finally:
            self.lock.release()
        return {"status": "retrieved_only", "answer": None, "generation_called": False,
                "top_k": k, "retrieved_count": len(hits), "context": packet,
                "school_level": school_level,
                "missing_date_conditions": missing_dates(packet),
                "condition_audit": condition_audit(packet),
                "elapsed_ms": round((time.perf_counter()-started)*1000),
                "retriever": self.info()}


class SearchServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, service):
        if address[0] != "127.0.0.1":
            raise ValueError("로컬 주소 127.0.0.1에서만 실행할 수 있습니다.")
        self.service = service
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(20)

    def log_message(self, format, *args):
        # No request/query logging: questions stay out of terminal history/logs.
        pass

    def allowed_request(self):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origin = self.headers.get("Origin")
        if self.headers.get("Host") not in hosts:
            self.send_json(403, {"error": "안내된 주소로 다시 접속해 주세요."})
            return False
        if origin is not None and origin not in {f"http://{host}" for host in hosts}:
            self.send_json(403, {"error": "다른 사이트에서의 요청은 허용하지 않습니다."})
            return False
        return True

    def send_body(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, payload):
        self.send_body(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def source_path(self, request_path):
        prefix, suffix = "/source/", ".pdf"
        if not request_path.startswith(prefix) or not request_path.endswith(suffix):
            return None
        encoded = request_path[len(prefix):-len(suffix)]
        try:
            doc_id = unquote(encoded, errors="strict")
        except UnicodeError:
            return None
        return self.server.service.source_path(doc_id)

    def do_GET(self):
        if not self.allowed_request():
            return
        request_path = urlsplit(self.path).path
        if request_path in STATIC:
            filename, content_type = STATIC[request_path]
            self.send_body(200, (WEB / filename).read_bytes(), content_type)
        elif request_path == "/api/info":
            self.send_json(200, self.server.service.info())
        elif request_path == "/favicon.ico":
            self.send_body(204, b"", "image/x-icon")
        elif request_path.startswith("/source/"):
            source = self.source_path(request_path)
            if source:
                self.send_body(200, source.read_bytes(), "application/pdf")
            else:
                self.send_json(404, {"error": "원문 파일을 찾을 수 없어요."})
        else:
            self.send_json(404, {"error": "요청한 경로가 없습니다."})

    def do_POST(self):
        if not self.allowed_request():
            return
        if self.path != "/api/search":
            self.send_json(404, {"error": "요청한 기능을 찾을 수 없어요."})
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self.send_json(415, {"error": "검색 요청을 읽지 못했어요. 화면을 새로고침한 뒤 다시 시도해 주세요."})
            return
        if self.headers.get("Transfer-Encoding"):
            self.send_json(400, {"error": "검색 요청을 읽지 못했어요. 다시 시도해 주세요."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_REQUEST_BYTES:
                self.send_json(413, {"error": "요청 크기가 허용 범위를 벗어났습니다."})
                return
            payload = json.loads(self.rfile.read(size))
            self.send_json(200, self.server.service.search(payload))
        except (ValueError, UnicodeError):
            self.send_json(400, {"error": "질문은 1~4000자, 검색 개수는 1~20으로 입력하세요."})
        except BusyError as exc:
            self.send_json(503, {"error": str(exc)})
        except Exception:
            self.send_json(500, {"error": "자료를 찾는 중에 문제가 생겼어요. 잠시 후 다시 시도해 주세요."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("포트는 1~65535여야 합니다.")
    print("로컬 검색 모델을 준비합니다. API 호출·모델 다운로드는 하지 않습니다.", flush=True)
    try:
        service = SearchService(DenseRetriever())
        server = SearchServer(("127.0.0.1", args.port), service)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"서버를 시작할 수 없습니다: {exc}\n")
    print(f"검색 화면: http://127.0.0.1:{server.server_port}  | 종료: Ctrl+C", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
