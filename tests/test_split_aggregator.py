import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation import SplitEvaluationAggregator
from src.io import JsonSaver


class TestSplitEvaluationAggregator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="split_agg_")
        self.exam_dir = os.path.join(self.temp_dir, "outputs", "evaluation", "exam_unit")
        self.split_root = os.path.join(self.temp_dir, "splits")
        os.makedirs(self.exam_dir, exist_ok=True)
        os.makedirs(self.split_root, exist_ok=True)

        JsonSaver().save(
            os.path.join(self.exam_dir, "overview_records.json"),
            [
                {"dataset_name": "summe", "video_id": "Jumps", "variant_name": "normalized_raw", "f1": 0.5, "precision": 0.6, "recall": 0.7, "rho": 0.4, "tau": 0.3},
                {"dataset_name": "summe", "video_id": "Cooking", "variant_name": "normalized_raw", "f1": 0.7, "precision": 0.8, "recall": 0.9, "rho": 0.6, "tau": 0.5},
                {"dataset_name": "summe", "video_id": "Jumps", "variant_name": "normalized_smoothed", "f1": 0.6, "precision": 0.7, "recall": 0.8, "rho": 0.5, "tau": 0.4},
                {"dataset_name": "summe", "video_id": "Cooking", "variant_name": "normalized_smoothed", "f1": 0.8, "precision": 0.9, "recall": 1.0, "rho": 0.7, "tau": 0.6},
            ],
        )
        JsonSaver().save(
            os.path.join(self.split_root, "summe_mapping.json"),
            {"video_1": "Jumps", "video_2": "Cooking"},
        )
        JsonSaver().save(
            os.path.join(self.split_root, "summe_splits_5.json"),
            [
                {"test_keys": ["video_1"], "train_keys": ["video_2"]},
                {"test_keys": ["video_2"], "train_keys": ["video_1"]},
            ],
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_aggregate_exam_dataset(self):
        aggregator = SplitEvaluationAggregator(split_root=self.split_root, split_count=5)
        summary = aggregator.aggregate_exam(self.exam_dir, ["summe"])

        self.assertTrue(os.path.exists(os.path.join(self.exam_dir, "split_overview.json")))
        self.assertTrue(os.path.exists(os.path.join(self.exam_dir, "split_overview.md")))

        dataset_summary = summary["datasets"]["summe"]
        raw = dataset_summary["variants"]["normalized_raw"]
        smooth = dataset_summary["variants"]["normalized_smoothed"]

        self.assertEqual(raw["num_splits"], 2)
        self.assertAlmostEqual(raw["mean_f1"], 0.6, places=6)
        self.assertAlmostEqual(raw["mean_precision"], 0.7, places=6)
        self.assertAlmostEqual(raw["mean_recall"], 0.8, places=6)
        self.assertAlmostEqual(smooth["mean_f1"], 0.7, places=6)
        self.assertEqual(raw["best_split"]["split_id"], 2)

        with open(os.path.join(self.exam_dir, "split_overview.json"), "r", encoding="utf-8") as file_obj:
            persisted = json.load(file_obj)
        self.assertIn("summe", persisted["datasets"])


if __name__ == "__main__":
    unittest.main()