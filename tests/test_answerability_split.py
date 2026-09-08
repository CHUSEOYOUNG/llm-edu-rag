import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_answerability_split import (evaluate_by_difficulty, read_benchmark,
                                           service_policy_metrics)


def row(qid, split, answerable, evidence):
    return {"qid": qid, "split": split, "answerable": answerable,
            "difficulty": "in_domain" if answerable else "hard_negative",
            "question": qid, "evidence_chunk_ids": evidence}


class AnswerabilitySplitTests(unittest.TestCase):
    def write(self, rows):
        temp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        with temp:
            temp.write("\n".join(json.dumps(item) for item in rows))
        self.addCleanup(Path(temp.name).unlink)
        return Path(temp.name)

    def valid_rows(self):
        return [row("d1", "development", True, ["c1"]),
                row("d2", "development", False, []),
                row("t1", "test", True, ["c1"]),
                row("t2", "test", False, [])]

    def test_benchmark_requires_evidence_for_answerable_questions(self):
        self.assertEqual(len(read_benchmark(self.write(self.valid_rows()), {"c1"})), 4)
        invalid = self.valid_rows()
        invalid[0]["evidence_chunk_ids"] = []
        with self.assertRaisesRegex(ValueError, "라벨과 근거"):
            read_benchmark(self.write(invalid), {"c1"})

    def test_benchmark_rejects_test_split_without_both_labels(self):
        invalid = self.valid_rows()[:-1]
        with self.assertRaisesRegex(ValueError, "test에 두 라벨"):
            read_benchmark(self.write(invalid), {"c1"})

    def test_difficulty_breakdown_counts_review_flags(self):
        rows = [{"difficulty": "in_domain", "top_score": .8},
                {"difficulty": "hard_negative", "top_score": .7},
                {"difficulty": "hard_negative", "top_score": .3}]
        result = evaluate_by_difficulty(rows, .6)
        self.assertEqual(result["in_domain"]["strong_candidate"], 1)
        self.assertEqual(result["hard_negative"]["review_recommended"], 1)

    def test_service_policy_applies_local_information_signal(self):
        rows = [
            {"answerable": True, "question": "중학교 수업 시간", "top_score": .8},
            {"answerable": False, "question": "우리 학교 전학 신청 마감일", "top_score": .8},
        ]
        result = service_policy_metrics(rows)
        self.assertEqual((result["tp"], result["tn"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
