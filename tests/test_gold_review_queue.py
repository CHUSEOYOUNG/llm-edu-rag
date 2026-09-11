from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from build_gold_review_queue import build_queue


class GoldReviewQueueTests(unittest.TestCase):
    def test_exact_duplicates_are_excluded_and_candidates_are_not_promoted(self):
        current = [{"question": "초등학교 교과는?", "gold_chunks": ["old"]}]
        candidates = [
            {"qid": "same", "question": "초등학교 교과는?", "answerable": True,
             "evidence_chunk_ids": ["old"], "split": "test", "difficulty": "in_domain"},
            {"qid": "new", "question": "출결에서 결과란?", "answerable": True,
             "evidence_chunk_ids": ["new"], "split": "test", "difficulty": "in_domain"},
        ]
        queue = build_queue(current, candidates)
        self.assertEqual([row["candidate_id"] for row in queue], ["new"])
        self.assertEqual(queue[0]["candidate_kind"], "new_evidence_candidate")
        self.assertEqual(queue[0]["annotation_status"], "pending_human_review")
        self.assertIsNone(queue[0]["reference_answer"])

    def test_overlap_and_unanswerable_candidates_are_labeled_separately(self):
        current = [{"question": "기존", "gold_chunks": ["old"]}]
        candidates = [
            {"qid": "para", "question": "다른 표현", "answerable": True,
             "evidence_chunk_ids": ["old"], "split": "development",
             "difficulty": "in_domain"},
            {"qid": "none", "question": "오늘 급식", "answerable": False,
             "evidence_chunk_ids": [], "split": "development",
             "difficulty": "hard_negative"},
        ]
        queue = build_queue(current, candidates)
        self.assertEqual([row["candidate_kind"] for row in queue], [
            "existing_evidence_paraphrase", "unanswerable_candidate"
        ])
        self.assertTrue(all(row["evaluation_role"] == "annotation_candidate_only"
                            for row in queue))


if __name__ == "__main__":
    unittest.main()
