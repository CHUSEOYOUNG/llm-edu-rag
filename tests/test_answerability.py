from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_answerability import best_threshold, metrics, roc_auc


class AnswerabilityBaselineTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"answerable": True, "top_score": .9},
            {"answerable": True, "top_score": .7},
            {"answerable": False, "top_score": .6},
            {"answerable": False, "top_score": .2},
        ]

    def test_metrics_use_answerable_as_positive_class(self):
        result = metrics(self.rows, .65)
        self.assertEqual({key: result[key] for key in ("tp", "fp", "tn", "fn")},
                         {"tp": 2, "fp": 0, "tn": 2, "fn": 0})
        self.assertEqual(result["balanced_accuracy"], 1.0)

    def test_auc_and_best_threshold_are_deterministic(self):
        self.assertEqual(roc_auc(self.rows), 1.0)
        threshold, result = best_threshold(self.rows)
        self.assertAlmostEqual(threshold, .65)
        self.assertEqual(result["accuracy"], 1.0)

    def test_single_class_input_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics([{"answerable": True, "top_score": .9}], .5)
        with self.assertRaises(ValueError):
            roc_auc([{"answerable": False, "top_score": .1}])


if __name__ == "__main__":
    unittest.main()
