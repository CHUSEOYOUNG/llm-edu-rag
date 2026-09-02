import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ablation_chunking import (evaluate_mapped_groups, evidence_options,
                               fixed_window_chunks, overlap_chunks,
                               structure_chunks)


def section(text="가" * 900):
    return {
        "doc_id": "문서", "path": "대단원", "number": "1", "title": "항목",
        "text": text, "line_pages": None,
    }


class ChunkingVariantTests(unittest.TestCase):
    def test_fixed_windows_keep_all_text_and_merge_a_short_tail(self):
        text = "가" * 1650
        chunks = fixed_window_chunks([section(text)], size=800, min_tail=100)
        self.assertEqual([chunk["n_chars"] for chunk in chunks], [800, 850])
        self.assertEqual("".join(chunk["body"] for chunk in chunks), text)
        self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), 2)

    def test_overlap_adds_the_previous_tail_without_changing_the_first_chunk(self):
        text = ("첫 문단 " * 100).strip() + "\n\n" + ("둘째 문단 " * 100).strip()
        base = structure_chunks([section(text)])
        chunks = overlap_chunks([section(text)], overlap=80)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["body"], base[0]["body"])
        self.assertTrue(chunks[1]["body"].endswith(base[1]["body"]))
        self.assertGreater(chunks[1]["n_chars"], base[1]["n_chars"])

    def test_evidence_is_remapped_by_document_and_exact_text(self):
        chunks = [
            {"chunk_id": "a", "doc_id": "문서", "body": "정답 문장", "path": "", "n_chars": 5},
            {"chunk_id": "b", "doc_id": "다른 문서", "body": "정답 문장", "path": "", "n_chars": 5},
        ]
        question = {"evidence_groups": [
            {"alternatives": [{"doc_id": "문서", "text": "정답 문장"}]},
            {"alternatives": [{"doc_id": "문서", "text": "경계에서 잘린 문장"}]},
        ]}
        self.assertEqual(evidence_options(question, chunks), [{"a"}, set()])

    def test_unmappable_groups_count_as_missing_not_as_perfect(self):
        metrics = evaluate_mapped_groups(["a", "noise"], [{"a"}, set()])
        self.assertEqual(metrics["hit@1"], 1.0)
        self.assertEqual(metrics["coverage@5"], 0.5)
        self.assertEqual(metrics["complete@5"], 0.0)


if __name__ == "__main__":
    unittest.main()
