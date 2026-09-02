"""Loopback-only FastAPI evidence browser with no paid generation routes.

Run: python src/search_app.py, then open http://127.0.0.1:8765.
"""

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
import re
import threading
import time
from typing import Any, Literal
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

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
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "testserver"}
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
    "img-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
)
DOCS_CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src https://cdn.jsdelivr.net; "
    "style-src https://cdn.jsdelivr.net; connect-src 'self'; img-src data: https:; "
    "base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
)


class BusyError(RuntimeError):
    pass


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(min_length=1, max_length=4000,
                          description="학교생활이나 교육과정에 관한 질문")
    top_k: int = Field(default=5, ge=1, le=20,
                       description="한 번에 받을 관련 내용 수")
    school_level: Literal["all", "elementary", "middle", "high"] = "all"


class RetrieverInfo(BaseModel):
    mode: str
    generation_enabled: bool
    model: str
    index_text: str
    short_keyword_rerank: bool
    chunk_count: int
    document_count: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    ready: Literal[True]
    mode: Literal["local_retrieval_only"]


class SearchResponse(BaseModel):
    status: Literal["retrieved_only"]
    answer: None
    generation_called: Literal[False]
    top_k: int
    retrieved_count: int
    context: dict[str, Any]
    school_level: str
    missing_date_conditions: list[str]
    condition_audit: list[dict[str, Any]]
    elapsed_ms: int
    retriever: RetrieverInfo


class ErrorResponse(BaseModel):
    error: str


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


def request_host(request):
    return request.headers.get("host", "").split(":", 1)[0].lower()


def add_security_headers(response, docs=False):
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        DOCS_CONTENT_SECURITY_POLICY if docs else CONTENT_SECURITY_POLICY)
    return response


def create_app(service=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.search_service = service or SearchService(DenseRetriever())
        yield
        app.state.search_service = None

    api = FastAPI(
        title="학교생활 교육 문서 검색 API",
        description="교육 문서에서 관련 원문과 PDF 페이지를 찾는 로컬 검색 API",
        version="1.0.0",
        lifespan=lifespan,
        redoc_url=None,
    )

    @api.middleware("http")
    async def local_security(request, call_next):
        host = request_host(request)
        if host not in ALLOWED_HOSTS:
            response = JSONResponse(
                status_code=403, content={"error": "안내된 주소로 다시 접속해 주세요."})
            return add_security_headers(response)
        origin = request.headers.get("origin")
        if origin is not None and origin != f"http://{request.headers.get('host', '')}":
            response = JSONResponse(
                status_code=403, content={"error": "다른 사이트에서의 요청은 허용하지 않습니다."})
            return add_security_headers(response)
        if request.method == "POST" and request.url.path == "/api/search":
            if request.headers.get("transfer-encoding"):
                response = JSONResponse(
                    status_code=400, content={"error": "검색 요청을 읽지 못했어요. 다시 시도해 주세요."})
                return add_security_headers(response)
            content_type = request.headers.get("content-type", "").split(";", 1)[0].strip()
            if content_type != "application/json":
                response = JSONResponse(
                    status_code=415,
                    content={"error": "검색 요청을 읽지 못했어요. 화면을 새로고침한 뒤 다시 시도해 주세요."})
                return add_security_headers(response)
            try:
                size = int(request.headers.get("content-length", "0"))
            except ValueError:
                size = 0
            if not 0 < size <= MAX_REQUEST_BYTES:
                response = JSONResponse(
                    status_code=413, content={"error": "요청 크기가 허용 범위를 벗어났습니다."})
                return add_security_headers(response)
        response = await call_next(request)
        return add_security_headers(response, request.url.path == "/docs")

    @api.exception_handler(RequestValidationError)
    async def validation_error(_request, _exc):
        return JSONResponse(
            status_code=400,
            content={"error": "질문은 1~4000자, 검색 개수는 1~20으로 입력하세요."},
        )

    @api.exception_handler(StarletteHTTPException)
    async def http_error(_request, exc):
        if exc.status_code == 404:
            return JSONResponse(status_code=404, content={"error": "요청한 경로가 없습니다."})
        return JSONResponse(status_code=exc.status_code, content={"error": "요청을 처리하지 못했습니다."})

    @api.get("/health", response_model=HealthResponse, tags=["운영"])
    def health():
        return {"status": "ok", "ready": True, "mode": "local_retrieval_only"}

    @api.get("/api/info", response_model=RetrieverInfo, tags=["검색"])
    def info(request: Request):
        return request.app.state.search_service.info()

    @api.post(
        "/api/search",
        response_model=SearchResponse,
        responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
        tags=["검색"],
    )
    def search(payload: SearchRequest, request: Request):
        try:
            return request.app.state.search_service.search(payload.model_dump())
        except BusyError as exc:
            return JSONResponse(status_code=503, content={"error": str(exc)})
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "질문은 1~4000자, 검색 개수는 1~20으로 입력하세요."})
        except Exception:
            return JSONResponse(
                status_code=500,
                content={"error": "자료를 찾는 중에 문제가 생겼어요. 잠시 후 다시 시도해 주세요."})

    @api.get("/source/{doc_id}.pdf", include_in_schema=False)
    def source_pdf(doc_id: str, request: Request):
        source = request.app.state.search_service.source_path(doc_id)
        if source is None:
            return JSONResponse(status_code=404, content={"error": "원문 파일을 찾을 수 없어요."})
        return FileResponse(source, media_type="application/pdf")

    def static_response(path, content_type):
        def serve_static():
            return FileResponse(path, media_type=content_type)
        return serve_static

    for route, (filename, media_type) in STATIC.items():
        api.add_api_route(
            route,
            static_response(WEB / filename, media_type),
            methods=["GET"],
            include_in_schema=False,
        )

    @api.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204, media_type="image/x-icon")

    return api


app = create_app()


def run_server(port=8765, host="127.0.0.1"):
    if host != "127.0.0.1":
        raise ValueError("로컬 주소 127.0.0.1에서만 실행할 수 있습니다.")
    if not 1 <= port <= 65535:
        raise ValueError("포트는 1~65535여야 합니다.")
    import uvicorn

    print("로컬 검색 모델을 준비합니다. API 호출·모델 다운로드는 하지 않습니다.", flush=True)
    print(f"검색 화면: http://{host}:{port}  | API 문서: http://{host}:{port}/docs", flush=True)
    uvicorn.run(app, host=host, port=port, access_log=False, log_level="warning")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        run_server(args.port)
    except ValueError as exc:
        parser.exit(1, f"서버를 시작할 수 없습니다: {exc}\n")


if __name__ == "__main__":
    main()
