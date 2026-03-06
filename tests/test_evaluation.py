import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation import EvaluationBenchmark
from src.evaluation.metrics import (
    coverage_score,
    diversity_score,
    kendall_correlation,
    latency_statistics,
    precision_recall_fscore,
    spearman_correlation,
)


class TestEvaluationStep13(unittest.TestCase):
    def setUp(self):
        self.predicted_scores = ["drop", "low", "medium", "high", "high", "low"]
        self.gt_scores = [0, 0, 1, 1, 1, 0]
        self.selected_segments = [
            {
                "start_frame": 2,
                "end_frame": 4,
                "summary_text": "busy downtown street with strong motion",
                "selection_reason": "主线推进明确",
            },
            {
                "start_frame": 4,
                "end_frame": 6,
                "summary_text": "arrival at a plaza with different activity",
                "selection_reason": "事件价值高",
            },
        ]
        self.temp_dir = tempfile.mkdtemp(prefix="eval_step13_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_metrics_functions(self):
        prf = precision_recall_fscore(self.predicted_scores, self.gt_scores)
        self.assertAlmostEqual(prf["precision"], 1.0, places=4)
        self.assertAlmostEqual(prf["recall"], 1.0, places=4)
        self.assertAlmostEqual(prf["fscore"], 1.0, places=4)

        self.assertGreaterEqual(spearman_correlation(self.predicted_scores, self.gt_scores), 0.8)
        self.assertGreaterEqual(kendall_correlation(self.predicted_scores, self.gt_scores), 0.6)
        self.assertAlmostEqual(coverage_score(self.predicted_scores, self.gt_scores), 1.0, places=4)
        self.assertGreater(diversity_score(self.selected_segments), 0.1)

        latency = latency_statistics([0.1, 0.2, 0.3, 0.4])
        self.assertEqual(latency["count"], 4)
        self.assertAlmostEqual(latency["mean_sec"], 0.25, places=4)

    def test_benchmark_report_and_plot(self):
        benchmark = EvaluationBenchmark()
        metrics = benchmark.evaluate(
            predicted_scores=self.predicted_scores,
            gt_scores=self.gt_scores,
            selected_segments=self.selected_segments,
            latencies_sec=[0.1, 0.15, 0.18],
        )
        artifacts = benchmark.save_report(
            metrics=metrics,
            predicted_scores=self.predicted_scores,
            gt_scores=self.gt_scores,
            output_dir=self.temp_dir,
            video_id="demo_eval",
        )

        self.assertTrue(os.path.exists(artifacts["metrics_path"]))
        self.assertTrue(os.path.exists(artifacts["plot_path"]))

        with open(artifacts["metrics_path"], "r", encoding="utf-8") as file_obj:
            saved = json.load(file_obj)

        self.assertIn("fscore", saved)
        self.assertIn("latency", saved)


if __name__ == "__main__":
    unittest.main()