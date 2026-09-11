from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from query_planning import (interleave_rankings, school_comparison_queries,
                            supplemental_content_query)


class QueryPlanningTests(unittest.TestCase):
    def test_scope_heavy_question_gets_content_query_without_losing_user_question(self):
        question = (
            "코퍼스에 포함된 국가교육위원회 고시 제2026-1호에서, "
            "2028년 3월 1일부터 초등학교 1·2학년에 적용하도록 정한 교과는 무엇인가요?"
        )
        self.assertEqual(
            supplemental_content_query(question),
            "초등학교 1·2학년에 교과는 무엇인가요?",
        )
        self.assertIn("2028년 3월 1일", question)

    def test_ordinary_question_is_not_rewritten(self):
        self.assertIsNone(supplemental_content_query("작년 생기부를 고칠 수 있나요?"))
        self.assertIsNone(supplemental_content_query("2028년 3월 1일은 무슨 요일인가요?"))
        self.assertIsNone(supplemental_content_query(
            "국가교육위원회 고시 제2026-1호에서 기준은?"
        ))

    def test_rankings_are_interleaved_and_deduplicated(self):
        core = [{"chunk_id": "answer"}, {"chunk_id": "shared"}]
        original = [{"chunk_id": "scope"}, {"chunk_id": "shared"}]
        self.assertEqual(
            [hit["chunk_id"] for hit in interleave_rankings([core, original], 4)],
            ["answer", "scope", "shared"],
        )

    def test_school_comparison_is_split_without_answer_words(self):
        question = (
            "2022 개정 교육과정 기준으로 초등학교와 중학교의 "
            "수업 한 시간은 각각 몇 분이 원칙인가요?"
        )
        self.assertEqual(school_comparison_queries(question), [
            {"school_level": "elementary",
             "query": "초등학교 교육과정 수업 한 시간 몇 분"},
            {"school_level": "middle",
             "query": "중학교 교육과정 수업 한 시간 몇 분"},
        ])

    def test_single_school_question_is_not_split(self):
        self.assertEqual(school_comparison_queries(
            "중학교 수업 한 시간은 몇 분인가요?"
        ), [])

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            supplemental_content_query(" ")
        with self.assertRaises(ValueError):
            school_comparison_queries(" ")
        with self.assertRaises(ValueError):
            interleave_rankings([], 5)
        with self.assertRaises(ValueError):
            interleave_rankings([[]], 0)


if __name__ == "__main__":
    unittest.main()
