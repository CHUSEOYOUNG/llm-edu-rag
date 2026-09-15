from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regression_eval import (citation_is_valid, evidence_groups, reference_numbers_match,
                             regression_failures, retrieval_metrics, run_regression)


class FakeService:
    def search(self, payload):
        sources = [{"source_id": "S1", "chunk_id": "gold", "body": "정답 문장",
                    "path": "조항", "doc_id": "문서", "score": .8}]
        return {"context": {"sources": sources}}

    def answer(self, payload):
        source = {"source_id": "S1", "chunk_id": "gold", "body": "정답 문장",
                  "path": "조항", "doc_id": "문서", "score": .8}
        return {"status": "draft_answer", "answer": "정답입니다. [S1]", "reason": "",
                "context": {"sources": [source]}, "claims": [{
                    "text": "정답입니다.", "evidence": [{"source_id": "S1",
                    "chunk_id": "gold", "start": 0, "end": 5, "quote": "정답 문장"}]}]}


class RegressionEvaluationTests(unittest.TestCase):
    def test_retrieval_scores_evidence_groups_not_document_variants(self):
        metrics = retrieval_metrics(["noise", "middle-copy"], [
            {"elementary-copy", "middle-copy"}
        ])
        self.assertEqual(metrics["recall@1"], 0)
        self.assertEqual(metrics["recall@5"], 1)
        self.assertEqual(metrics["complete@5"], 1)
        self.assertEqual(metrics["mrr@10"], .5)

    def test_literal_citation_validation_checks_the_original_span(self):
        question = {"qid": "q1", "type": "fact", "question": "질문",
                    "gold_chunks": ["gold"], "meta": None}
        result = run_regression(FakeService(), [question], generate=True)
        self.assertTrue(result["generation"]["per_question"][0]["citation_ok"])
        answer = FakeService().answer({})
        answer["claims"][0]["evidence"][0]["quote"] = "없는 문장"
        self.assertFalse(citation_is_valid(answer))

    def test_v2_groups_take_precedence_over_legacy_gold(self):
        groups = evidence_groups({"gold_chunks": ["old"], "evidence_groups": [
            {"alternatives": [{"chunk_id": "new-a"}, {"chunk_id": "new-b"}]}
        ]})
        self.assertEqual(groups, [{"new-a", "new-b"}])

    def test_baseline_comparison_fails_only_on_lower_core_metrics(self):
        def score(recall):
            return {"retrieval": {"overall": {"recall@5": recall,
                    "complete@5": recall, "mrr@10": recall}},
                    "generation": {"enabled": False}}
        self.assertEqual(regression_failures(score(.8), score(.8)), [])
        self.assertIn("retrieval.overall.recall@5: 0.8000 -> 0.7000",
                      regression_failures(score(.7), score(.8)))

    def test_follow_up_passes_previous_question_and_reports_resolved_query(self):
        service = FakeService()
        original_search = service.search
        service.search = Mock(side_effect=original_search)
        question = {
            "qid": "f1", "type": "fact", "primary_category": "follow_up",
            "question": "그럼 중학교는?", "previous_question": "초등학교는 몇 분인가요?",
            "expected_resolved_question": "중학교는 몇 분인가요?",
            "evidence_groups": [{"alternatives": [{"chunk_id": "gold"}]}],
        }
        result = run_regression(service, [question])
        self.assertEqual(service.search.call_args.args[0]["previous_question"],
                         "초등학교는 몇 분인가요?")
        self.assertEqual(result["retrieval"]["by_category"]["follow_up"]["evaluated_count"], 1)

    def test_pending_or_sealed_questions_are_rejected(self):
        base = {"qid": "q", "question": "질문", "type": "fact"}
        for change in ({"annotation_status": "pending_human_review"},
                       {"release_status": "sealed"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                run_regression(FakeService(), [{**base, **change}])

    def test_reference_number_match_detects_wrong_table_total(self):
        self.assertTrue(reference_numbers_match("2년간 204시간", "2년간 204시간입니다."))
        self.assertFalse(reference_numbers_match("2년간 204시간", "2년간 2,176시간입니다."))
        self.assertIsNone(reference_numbers_match("과목 목록", "국어와 수학입니다."))

    def test_progress_reports_each_question_index(self):
        events = []
        questions = [
            {"qid": qid, "type": "fact", "question": qid,
             "evidence_groups": [{"alternatives": [{"chunk_id": "gold"}]}]}
            for qid in ("q1", "q2")
        ]
        run_regression(FakeService(), questions,
                       progress=lambda *event: events.append(event))
        self.assertEqual(events, [
            ("search", 1, 2, "q1"),
            ("search", 2, 2, "q2"),
        ])


if __name__ == "__main__":
    unittest.main()
