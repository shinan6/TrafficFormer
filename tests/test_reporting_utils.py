import json
import tempfile
import unittest
from pathlib import Path

from uer.reporting_utils import (
    compute_metrics,
    write_metrics_json,
    write_confusion_matrix_csv,
    write_predictions_tsv,
    aggregate_fold_metrics,
)


class TestComputeMetrics(unittest.TestCase):
    def test_perfect_predictions(self):
        m = compute_metrics([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2], ["A", "B", "C"])
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["macro_f1"], 1.0)

    def test_imperfect_predictions(self):
        m = compute_metrics([0, 0, 1, 1], [0, 1, 1, 0], ["A", "B"])
        self.assertAlmostEqual(m["accuracy"], 0.5)
        self.assertIn("per_class", m)

    def test_returns_all_expected_keys(self):
        m = compute_metrics([0, 1], [0, 1], ["A", "B"])
        for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]:
            self.assertIn(key, m)


class TestWriteFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_metrics_json_roundtrips(self):
        metrics = {"accuracy": 0.95, "macro_f1": 0.9}
        write_metrics_json(self.d / "m.json", metrics)
        loaded = json.loads((self.d / "m.json").read_text())
        self.assertEqual(loaded["accuracy"], 0.95)

    def test_write_confusion_matrix_has_labels(self):
        write_confusion_matrix_csv(self.d / "cm.csv", [0, 0, 1, 1], [0, 1, 1, 0], ["A", "B"])
        text = (self.d / "cm.csv").read_text()
        self.assertIn("A", text)
        self.assertIn("B", text)
        lines = text.strip().split("\n")
        self.assertEqual(len(lines), 3)

    def test_write_predictions_tsv(self):
        write_predictions_tsv(self.d / "p.tsv", [0, 1, 0], [0, 0, 1], ["A", "B"])
        lines = (self.d / "p.tsv").read_text().strip().split("\n")
        self.assertEqual(len(lines), 4)
        self.assertIn("true_label", lines[0])


class TestAggregation(unittest.TestCase):
    def test_aggregate_computes_mean_and_std(self):
        folds = [
            {"accuracy": 0.8, "macro_precision": 0.7, "macro_recall": 0.75,
             "macro_f1": 0.72, "weighted_f1": 0.78},
            {"accuracy": 0.9, "macro_precision": 0.85, "macro_recall": 0.88,
             "macro_f1": 0.86, "weighted_f1": 0.89},
        ]
        agg = aggregate_fold_metrics(folds)
        self.assertAlmostEqual(agg["accuracy"]["mean"], 0.85)
        self.assertEqual(len(agg["accuracy"]["per_fold"]), 2)
        self.assertGreater(agg["accuracy"]["std"], 0)

    def test_aggregate_empty_returns_empty(self):
        self.assertEqual(aggregate_fold_metrics([]), {})


if __name__ == "__main__":
    unittest.main()
