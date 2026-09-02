from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import re
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from search_app import (BusyError, SearchService, condition_audit, create_app,
                        keyword_rerank, keyword_terms, matches_school_level,
                        run_server)
from rag import build_packet
from test_rag import hit


def service():
    retriever = Mock()
    retriever.config = {"model": "test-model", "index_text": "body"}
    retriever.chunks = [hit()]
    retriever.search.return_value = [hit()]
    return SearchService(retriever)


class SearchServiceTests(unittest.TestCase):
    def test_short_keywords_prefer_matching_section_titles_over_dense_mentions(self):
        dense_first = hit("dense", body="정정 사례의 출결상황 입력 누락", path="자료의 정정")
        overview = hit("overview", body="수업일수와 결석일수를 입력한다.", path="8조 출결상황")
        middle = hit("middle", body="출결 처리 안내", path="8조 출결상황",
                     doc_id="2026 학교생활기록부 기재요령(중)_F_260227")
        self.assertEqual(keyword_terms("출결"), ["출결"])
        self.assertEqual(keyword_terms("중학교 출결"), ["중학교", "출결"])
        self.assertEqual(keyword_terms("출결은 어떻게 처리하나요?"), [])
        self.assertEqual(keyword_rerank("출결", [dense_first, overview])[0]["chunk_id"], "overview")
        self.assertEqual(keyword_rerank("중학교 출결", [overview, middle])[0]["chunk_id"], "middle")

    def test_short_keyword_search_expands_candidates_before_reranking(self):
        retriever = Mock()
        retriever.config = {"model": "test-model", "index_text": "body"}
        retriever.chunks = [hit(f"c{i}") for i in range(150)]
        retriever.search.return_value = [
            hit("dense", body="출결 입력 누락", path="자료의 정정"),
            hit("overview", body="결석일수 안내", path="8조 출결상황"),
        ]
        result = SearchService(retriever).search({"question": "출결", "top_k": 5})
        retriever.search.assert_called_once_with("출결", 100)
        self.assertEqual(result["context"]["sources"][0]["chunk_id"], "overview")

    def test_school_filter_keeps_specific_and_explicit_general_material(self):
        elementary = hit("e", doc_id="2026 학교생활기록부 기재요령(초)_F_260219")
        middle = hit("m", doc_id="2026 학교생활기록부 기재요령(중)_F_260227")
        general_middle = hit("g", body="중학교 수업은 45분을 원칙으로 한다.", doc_id="교육과정 총론")
        unrelated_general = hit("x", body="공통 운영 안내", doc_id="교육과정 총론")
        self.assertTrue(matches_school_level(middle, "middle"))
        self.assertTrue(matches_school_level(general_middle, "middle"))
        self.assertFalse(matches_school_level(elementary, "middle"))
        self.assertFalse(matches_school_level(
            hit("cross", body="고등학교와 함께 보는 안내", doc_id=middle["doc_id"]), "high"))
        self.assertFalse(matches_school_level(unrelated_general, "middle"))

        retriever = Mock()
        retriever.config = {"model": "test-model", "index_text": "body"}
        retriever.chunks = [elementary, middle, general_middle, unrelated_general]
        retriever.search.return_value = retriever.chunks
        result = SearchService(retriever).search(
            {"question": "수업 시간", "top_k": 5, "school_level": "middle"})
        retriever.search.assert_called_once_with("수업 시간", 4)
        self.assertEqual([source["chunk_id"] for source in result["context"]["sources"]], ["m", "g"])
        self.assertEqual(result["school_level"], "middle")
        self.assertTrue(result["context"]["scope_filters_applied"])

    def test_search_preserves_original_question_and_never_generates(self):
        search = service()
        question = "한글 한 글자는 몇 바이트인가요?"
        with patch("rag_generate.generate", side_effect=AssertionError("API forbidden")):
            result = search.search({"question": question, "top_k": 5})
        search.retriever.search.assert_called_once_with(question, 5)
        self.assertEqual(result["context"]["original_question"], question)
        self.assertFalse(result["generation_called"])
        self.assertIsNone(result["answer"])
        self.assertEqual(result["context"]["sources"][0]["body"], hit()["body"])

    def test_invalid_payloads_never_reach_retrieval(self):
        search = service()
        for payload in (None, {}, [], {"question": ""}, {"question": "x"*4001},
                        {"question": "질문", "top_k": True}, {"question": "질문", "top_k": 21},
                        {"question": "질문", "top_k": 1.5}, {"question": "질문", "generate": True},
                        {"question": "질문", "school_level": "university"}):
            with self.subTest(payload=str(payload)[:40]):
                with self.assertRaises(ValueError): search.search(payload)
        search.retriever.search.assert_not_called()

    def test_simultaneous_model_use_is_rejected_and_lock_released_after_error(self):
        search = service()
        search.lock.acquire()
        try:
            with self.assertRaises(BusyError): search.search({"question": "질문"})
        finally:
            search.lock.release()
        search.retriever.search.side_effect = RuntimeError("model failed")
        with self.assertRaises(RuntimeError): search.search({"question": "질문"})
        self.assertFalse(search.lock.locked())

    def test_literal_scope_audit_distinguishes_body_path_and_filename(self):
        packet = build_packet("2028년 3월 1일 초등학교 교과는?", [
            hit(body="초등학교 교과 목록", path="", doc_id="2028년 3월 1일 문서")])
        audit = condition_audit(packet)
        self.assertEqual(audit[0]["sources"][0]["fields"], ["doc_id"])
        self.assertEqual(audit[0]["semantic_verdict"], "not_verified")
        self.assertEqual(audit[1]["sources"][0]["fields"], ["body"])

    def test_only_existing_local_pdf_gets_a_page_link(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "문서.pdf"
            source.write_bytes(b"%PDF-1.7 test")
            retriever = Mock()
            retriever.config = {"model": "test-model", "index_text": "body"}
            retriever.chunks = [{**hit(), "page_start": 7, "page_end": 8}]
            retriever.search.return_value = retriever.chunks
            search = SearchService(retriever, Path(temp))
            result = search.search({"question": "질문"})
            found = result["context"]["sources"][0]
            self.assertEqual((found["page_start"], found["page_end"]), (7, 8))
            self.assertEqual(found["source_url"], "/source/%EB%AC%B8%EC%84%9C.pdf#page=7")
            self.assertEqual(search.source_path("문서"), source.resolve())


class SearchHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = service()
        cls.client_context = TestClient(create_app(cls.service))
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def request(self, method, path, body=None, headers=None):
        response = self.client.request(method, path, content=body, headers=headers or {})
        return response.status_code, response.headers, response.content

    def test_static_assets_and_info_are_served_without_external_resources(self):
        for path in ("/", "/app.css", "/app.css?v=20260901-3", "/presentation.js",
                     "/app.js", "/api/info"):
            with self.subTest(path=path):
                status, headers, body = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertIn("default-src 'none'", headers["Content-Security-Policy"])
                self.assertEqual(headers["Cache-Control"], "no-store")
                self.assertTrue(body)
        _, _, body = self.request("GET", "/api/info")
        self.assertFalse(json.loads(body)["generation_enabled"])

    def test_health_and_openapi_document_the_search_contract(self):
        response = self.client.get("/health")
        self.assertEqual(response.json(), {
            "status": "ok", "ready": True, "mode": "local_retrieval_only"})
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/api/search", schema["paths"])
        self.assertIn("SearchRequest", schema["components"]["schemas"])
        request_schema = schema["components"]["schemas"]["SearchRequest"]
        self.assertEqual(request_schema["additionalProperties"], False)
        docs = self.client.get("/docs")
        self.assertEqual(docs.status_code, 200)
        self.assertIn("cdn.jsdelivr.net", docs.headers["Content-Security-Policy"])

    def test_search_returns_full_source_and_no_generated_answer(self):
        status, _, body = self.request("POST", "/api/search", json.dumps({"question": "한글?"}),
                                       {"Content-Type": "application/json"})
        result = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "retrieved_only")
        self.assertEqual(result["context"]["sources"][0]["body"], hit()["body"])

    def test_filesystem_paths_and_generation_routes_are_not_exposed(self):
        for method, path in (("GET", "/.env"), ("GET", "/../README.md"), ("GET", "/src/rag.py"),
                             ("POST", "/api/generate")):
            with self.subTest(path=path):
                self.assertEqual(self.request(method, path)[0], 404)

    def test_foreign_hosts_and_origins_are_rejected(self):
        for headers in ({"Host": "evil.example"}, {"Origin": "https://evil.example"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.request("GET", "/api/info", headers=headers)[0], 403)

    def test_invalid_json_content_type_and_large_body_are_rejected(self):
        cases = [("not-json", {"Content-Type": "application/json"}, 400),
                 ("{}", {"Content-Type": "text/plain"}, 415),
                 ("x"*16385, {"Content-Type": "application/json"}, 413)]
        for body, headers, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(self.request("POST", "/api/search", body, headers)[0], expected)

    def test_request_schema_rejects_extra_and_coerced_fields(self):
        invalid = [
            {"question": "출결", "unknown": "field"},
            {"question": "출결", "top_k": True},
            {"question": "출결", "school_level": "university"},
        ]
        calls_before = self.service.retriever.search.call_count
        for payload in invalid:
            with self.subTest(payload=payload):
                response = self.client.post("/api/search", json=payload)
                self.assertEqual(response.status_code, 400)
        self.assertEqual(self.service.retriever.search.call_count, calls_before)

    def test_server_cannot_bind_public_interface(self):
        with self.assertRaises(ValueError):
            run_server(8765, host="0.0.0.0")


class SourcePdfHttpTests(unittest.TestCase):
    def test_whitelisted_pdf_is_served_and_other_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "문서.pdf"
            source.write_bytes(b"%PDF-1.7 local-source")
            retriever = Mock()
            retriever.config = {"model": "test-model", "index_text": "body"}
            retriever.chunks = [hit()]
            service_with_pdf = SearchService(retriever, Path(temp))
            with TestClient(create_app(service_with_pdf)) as client:
                response = client.get("/source/%EB%AC%B8%EC%84%9C.pdf")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["Content-Type"], "application/pdf")
                self.assertEqual(response.content, source.read_bytes())

                response = client.get("/source/..%2F.env.pdf")
                self.assertEqual(response.status_code, 404)


class EducationPageTests(unittest.TestCase):
    def test_script_references_existing_elements_and_labels_avoid_developer_terms(self):
        class Page(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids = []
                self.text = []
            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if 'id' in attrs:
                    self.ids.append(attrs['id'])
            def handle_data(self, data):
                self.text.append(data)
        root = Path(__file__).resolve().parents[1] / 'web'
        page = Page()
        page.feed((root/'index.html').read_text())
        ids = set(page.ids)
        self.assertEqual(len(ids), len(page.ids), 'duplicate element IDs')
        used = set(re.findall(r'\$\("([\w-]+)"\)', (root/'app.js').read_text()))
        self.assertEqual(used - ids, set(), 'script references a removed element')
        visible = ' '.join(page.text)
        for term in ('청크', '근거', '유사도', '코퍼스', '컨텍스트', 'JSON', 'BGE-M3', 'EVIDENCE'):
            with self.subTest(term=term):
                self.assertNotIn(term, visible)
        self.assertIn('자동으로 작성한 답변이 아니에요', visible)


if __name__ == "__main__":
    unittest.main()
