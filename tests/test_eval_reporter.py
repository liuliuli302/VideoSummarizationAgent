import os
import shutil
import sys
import tempfile
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data import DatasetVideoRecord, EvalResult, EvalVariantResult
from src.evaluation import EvaluationReporter


class TestEvaluationReporter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="eval_reporter_")
        self.reporter = EvaluationReporter(output_root=self.temp_dir, exam_name="exam_unit")
        self.record = DatasetVideoRecord(
            dataset_name="summe",
            video_id="demo_video.mp4",
            video_path="/tmp/demo_video.mp4",
            n_frames=6,
            picks=[0, 1, 2, 3, 4, 5],
            change_points=[[0, 2], [3, 5]],
            n_frame_per_seg=[3, 3],
            user_summary=[[0, 0, 1, 1, 1, 0]],
            user_scores=[[0.0, 0.1, 0.8, 0.9, 0.7, 0.2]],
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_normalize_and_smooth_scores(self):
        normalized = self.reporter.normalize_scores([2.0, 4.0, 6.0])
        self.assertEqual(normalized, [0.0, 0.5, 1.0])

        smoothed = self.reporter.smooth_scores([0.0, 0.0, 1.0, 0.0, 0.0], window_size=3)
        self.assertEqual(len(smoothed), 5)
        self.assertEqual(smoothed[0], 0.0)
        self.assertEqual(smoothed[4], 0.0)
        self.assertGreater(smoothed[1], smoothed[0])
        self.assertAlmostEqual(smoothed[1], smoothed[2], places=6)
        self.assertAlmostEqual(smoothed[2], smoothed[3], places=6)

    def test_save_video_report_and_overview(self):
        variants = [
            EvalVariantResult(
                variant_name="normalized_raw",
                frame_scores=[0.0, 0.1, 0.8, 0.9, 0.7, 0.2],
                eval_result=EvalResult(
                    dataset_name="summe",
                    video_id="demo_video.mp4",
                    f1=0.5,
                    precision=0.6,
                    recall=0.7,
                    rho=0.8,
                    tau=0.4,
                    selected_summary=[0, 0, 1, 1, 0, 0],
                ),
            ),
            EvalVariantResult(
                variant_name="normalized_smoothed",
                frame_scores=[0.1, 0.2, 0.7, 0.8, 0.6, 0.3],
                eval_result=EvalResult(
                    dataset_name="summe",
                    video_id="demo_video.mp4",
                    f1=0.55,
                    precision=0.65,
                    recall=0.75,
                    rho=0.82,
                    tau=0.43,
                    selected_summary=[0, 0, 1, 1, 0, 0],
                ),
            ),
        ]

        artifacts = self.reporter.save_video_report(
            record=self.record,
            variants=variants,
            original_frame_scores=[0.3, 0.4, 0.6, 0.8, 0.7, 0.5],
            smooth_window_size=5,
        )

        self.assertTrue(os.path.exists(artifacts["video_dir"]))
        self.assertTrue(os.path.exists(artifacts["plot_path"]))
        self.assertTrue(os.path.exists(artifacts["overview_path"]))
        self.assertTrue(os.path.exists(artifacts["overview_markdown_path"]))
        self.assertTrue(os.path.exists(os.path.join(artifacts["video_dir"], "eval_normalized_raw.json")))
        self.assertTrue(os.path.exists(os.path.join(artifacts["video_dir"], "eval_normalized_smoothed.json")))


if __name__ == "__main__":
    unittest.main()