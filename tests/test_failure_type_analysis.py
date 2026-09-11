from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from analyze_failure_types import run, validate_taxonomy


class FailureTypeAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.questions = [
            {"qid": "q1"}, {"qid": "q2"},
        ]
        self.taxonomy = {
            "schema_version": 1,
            "status": "draft",
            "description": "test",
            "rows": [
                {"qid": "q1", "family": "policy_lookup",
                 "evidence_layout": "single_passage", "operations": ["lookup"],
                 "rationale": "single"},
                {"qid": "q2", "family": "corpus_abstention",
                 "evidence_layout": "none", "operations": ["abstention"],
                 "rationale": "none"},
            ],
        }

    def test_taxonomy_must_cover_each_question_once(self):
        self.assertEqual(set(validate_taxonomy(self.taxonomy, self.questions)), {"q1", "q2"})
        self.taxonomy["rows"].pop()
        with self.assertRaises(ValueError):
            validate_taxonomy(self.taxonomy, self.questions)

    def test_metrics_are_aggregated_by_layout(self):
        regression = {
            "schema_version": 1,
            "retrieval": {"per_question": [
                {"qid": "q1", "metrics": {"complete@5": 1.0, "mrr@10": 0.5}},
            ]},
            "generation": {"per_question": [
                {"qid": "q1", "answerable": True, "actual_status": "draft_answer",
                 "task_success": True, "citation_ok": True},
                {"qid": "q2", "answerable": False,
                 "actual_status": "insufficient_evidence", "task_success": True,
                 "citation_ok": None},
            ]},
        }
        result = run(self.taxonomy, self.questions, regression)
        single = result["dimensions"]["evidence_layout"]["single_passage"]
        none = result["dimensions"]["evidence_layout"]["none"]
        self.assertEqual(single["retrieval"]["metrics"]["complete@5"], 1.0)
        self.assertEqual(single["generation"]["answer_success_rate"], 1.0)
        self.assertEqual(none["generation"]["unanswerable_rejection_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
