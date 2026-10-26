import unittest
from evaluation.metrics import calculate_metrics
from evaluation.io_utils import align_predictions


class MetricsTests(unittest.TestCase):
    def test_binary_metrics(self):
        rows = [
            {"id": 1, "label": 0, "prediction": 0},
            {"id": 2, "label": 0, "prediction": 1},
            {"id": 3, "label": 1, "prediction": 1},
            {"id": 4, "label": 1, "prediction": 0},
        ]
        m = calculate_metrics(rows)
        self.assertEqual(m["n_samples"], 4)
        self.assertEqual(m["correct"], 2)
        self.assertEqual(m["confusion_matrix"]["rows_true_cols_pred"], [[1, 1], [1, 1]])
        self.assertAlmostEqual(m["accuracy"], 0.5)
        self.assertAlmostEqual(m["macro_f1"], 0.5)

    def test_invalid_prediction_counted_wrong(self):
        rows = [{"id": 1, "label": 1, "prediction": None}]
        m = calculate_metrics(rows)
        self.assertEqual(m["invalid_predictions"], 1)
        self.assertEqual(m["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
