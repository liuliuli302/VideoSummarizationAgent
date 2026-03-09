import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.official_protocol import (  # noqa: E402
    build_summary_from_segments,
    evaluate_benchmark_video,
    evaluate_summe,
    evaluate_tvsum,
)


class TestOfficialProtocol(unittest.TestCase):
    def test_build_summary_from_segments_uses_budgeted_selection(self):
        frame_scores = [0.9, 0.8, 0.2, 0.1]
        summary = build_summary_from_segments(
            frame_scores=frame_scores,
            segments=[(0, 2), (2, 4)],
            n_frames=4,
            budget_ratio=0.5,
        )
        self.assertEqual(summary.tolist(), [1, 1, 0, 0])

    def test_summe_fscore_matches_mean_pairwise_definition(self):
        machine_summary = [1, 1, 0, 0]
        user_summary = np.asarray(
            [
                [1, 1, 0, 0],
                [1, 0, 1, 0],
            ],
            dtype=np.float32,
        )
        metrics = evaluate_summe(machine_summary, user_summary)
        self.assertAlmostEqual(metrics["precision"], 0.75, places=4)
        self.assertAlmostEqual(metrics["recall"], 0.75, places=4)
        self.assertAlmostEqual(metrics["fscore"], 0.75, places=4)

    def test_tvsum_fscore_uses_official_segment_conversion(self):
        machine_summary = [1, 1, 0, 0]
        user_scores = np.asarray(
            [
                [5, 1],
                [5, 1],
                [1, 5],
                [1, 5],
            ],
            dtype=np.float32,
        )
        metrics = evaluate_tvsum(
            machine_summary=machine_summary,
            user_scores=user_scores,
            segments=[(0, 2), (2, 4)],
            n_frames=4,
            budget_ratio=0.5,
        )
        self.assertAlmostEqual(metrics["precision"], 0.5, places=4)
        self.assertAlmostEqual(metrics["recall"], 0.5, places=4)
        self.assertAlmostEqual(metrics["fscore"], 0.5, places=4)

    def test_dispatch_supports_summe(self):
        report = evaluate_benchmark_video(
            dataset_name="SumMe",
            predicted_scores=[1.0, 1.0, 0.0, 0.0],
            scene_ranges=[{"start_frame": 0, "end_frame": 2}, {"start_frame": 2, "end_frame": 4}],
            n_frames=4,
            budget_ratio=0.5,
            user_summary=np.asarray([[1, 1, 0, 0]], dtype=np.float32),
        )
        self.assertEqual(report["machine_summary"], [1, 1, 0, 0])
        self.assertAlmostEqual(report["metrics"]["fscore"], 1.0, places=4)


if __name__ == "__main__":
    unittest.main()