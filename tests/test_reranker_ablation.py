import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ablation_reranker import first_evidence_ranks, latency_summary, stable_rerank


class RerankerAblationTests(unittest.TestCase):
    def test_rerank_is_restricted_to_candidates_and_keeps_ties_stable(self):
        self.assertEqual(
            stable_rerank(["dense-1", "dense-2", "dense-3"], [0.2, 0.9, 0.9]),
            ["dense-2", "dense-3", "dense-1"],
        )

    def test_invalid_candidates_or_scores_are_rejected(self):
        invalid = [
            ([], []),
            (["a"], []),
            (["a", "a"], [1, 2]),
            (["a"], [math.nan]),
            (["a"], [math.inf]),
        ]
        for candidates, scores in invalid:
            with self.subTest(candidates=candidates, scores=scores):
                with self.assertRaises(ValueError):
                    stable_rerank(candidates, scores)

    def test_first_evidence_rank_handles_alternatives_and_missing_groups(self):
        groups = [
            {"group_id": "one", "alternatives": [{"chunk_id": "a"}, {"chunk_id": "b"}]},
            {"group_id": "two", "alternatives": [{"chunk_id": "missing"}]},
        ]
        self.assertEqual(first_evidence_ranks(["noise", "b", "a"], groups),
                         {"one": 2, "two": None})

    def test_latency_summary_uses_nearest_rank_for_p95(self):
        summary = latency_summary(list(range(1, 21)))
        self.assertEqual(summary["total_ms"], 210)
        self.assertEqual(summary["median_ms_per_question"], 10.5)
        self.assertEqual(summary["p95_ms_per_question"], 19)


if __name__ == "__main__":
    unittest.main()
