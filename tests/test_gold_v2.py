import copy
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from build_gold_v2 import PLAN, build  # noqa: E402
from build_gold_v2_development import NEW_DEVELOPMENT_IDS, build as build_development  # noqa: E402
from validate_gold_v2 import validate  # noqa: E402


def corpus_chunk_ids() -> set[str] | None:
    path = ROOT / "data/processed/chunks.jsonl"
    if not path.exists():
        return None
    return {
        json.loads(line)["chunk_id"]
        for line in path.read_text().splitlines()
        if line.strip()
    }


class GoldV2Tests(unittest.TestCase):
    def test_seed_has_exact_quota_and_valid_evidence_ids(self):
        summary = validate(PLAN, build(), corpus_chunk_ids())
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["question_count"], 50)
        self.assertEqual(summary["split_counts"], {"development": 35, "test": 15})
        self.assertEqual(summary["pending_human_review"], 37)

    def test_pending_test_question_cannot_be_made_scoreable(self):
        rows = build()
        target = next(row for row in rows if row["split"] == "test")
        target["evaluation_role"] = "scoreable_test"
        summary = validate(PLAN, rows)
        self.assertFalse(summary["valid"])
        self.assertTrue(any("평가 대상" in error for error in summary["errors"]))

    def test_same_fact_cannot_cross_development_and_test(self):
        rows = build()
        development = next(row for row in rows if row["split"] == "development")
        test = next(row for row in rows if row["split"] == "test")
        test["fact_key"] = development["fact_key"]
        summary = validate(PLAN, rows)
        self.assertFalse(summary["valid"])
        self.assertTrue(any("동일 fact_key" in error for error in summary["errors"]))

    def test_existing_evidence_paraphrase_is_rejected_from_test(self):
        rows = copy.deepcopy(build())
        target = next(row for row in rows if row["split"] == "test")
        target["source_origin"] = "existing_evidence_paraphrase"
        summary = validate(PLAN, rows)
        self.assertFalse(summary["valid"])
        self.assertTrue(any("단순 말바꾸기" in error for error in summary["errors"]))

    @unittest.skipUnless(
        (ROOT / "data/processed/chunks.jsonl").exists(),
        "data/processed/chunks.jsonl은 저장소에 포함하지 않으므로 로컬에서만 검증한다",
    )
    def test_development_export_excludes_the_sealed_test_split(self):
        rows, manifest = build_development()
        self.assertEqual(len(rows), 35)
        self.assertEqual(manifest["new_question_count"], len(NEW_DEVELOPMENT_IDS))
        self.assertTrue(all(row["annotation_status"] == "assistant_reviewed_draft"
                            for row in rows))
        self.assertTrue(all(row["source_id"] not in {
            candidate["source_id"] for candidate in build() if candidate["split"] == "test"
        } for row in rows))
        self.assertEqual(sum(row["type"] == "unans" for row in rows), 8)


if __name__ == "__main__":
    unittest.main()
