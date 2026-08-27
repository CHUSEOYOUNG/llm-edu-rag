import http.client
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import re
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from search_app import BusyError, SearchServer, SearchService, condition_audit
from rag import build_packet
from test_rag import hit


def service():
    retriever = Mock()
    retriever.config = {"model": "test-model", "index_text": "body"}
    retriever.chunks = [hit()]
    retriever.search.return_value = [hit()]
    return SearchService(retriever)


class SearchServiceTests(unittest.TestCase):
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
                        {"question": "질문", "top_k": 1.5}, {"question": "질문", "generate": True}):
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


class SearchHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = service()
        cls.server = SearchServer(("127.0.0.1", 0), cls.service)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_static_assets_and_info_are_served_without_external_resources(self):
        for path in ("/", "/app.css", "/presentation.js", "/app.js", "/api/info"):
            with self.subTest(path=path):
                status, headers, body = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertIn("default-src 'none'", headers["Content-Security-Policy"])
                self.assertEqual(headers["Cache-Control"], "no-store")
                self.assertTrue(body)
        _, _, body = self.request("GET", "/api/info")
        self.assertFalse(json.loads(body)["generation_enabled"])

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

    def test_server_cannot_bind_public_interface(self):
        with self.assertRaises(ValueError): SearchServer(("0.0.0.0", 0), self.service)


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
