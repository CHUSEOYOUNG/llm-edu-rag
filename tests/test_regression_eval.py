import unittest

from regression_eval import (citation_is_valid, evidence_groups, regression_failures,
                             retrieval_metrics, run_regression)


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


if __name__ == "__main__":
    unittest.main()
