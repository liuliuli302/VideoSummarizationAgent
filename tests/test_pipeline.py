import json
import os
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import VideoSummaryInferenceEngine


class TestPipelineStep12(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="video_step12_")
        self.video_path = os.path.join(self.temp_dir, "pipeline_demo.avi")
        self.output_path = os.path.join(self.temp_dir, "summary.json")
        self._create_test_video(self.video_path, num_frames=18, fps=6)
        self.config = {
            "segment": {"coarse_segment_sec": 1.5},
            "window": {"win_len_sec": 1.0, "overlap_sec": 0.5, "sample_rate": 1},
            "summary": {"budget_ratio": 0.5},
            "memory": {"topk": 3},
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_test_video(self, video_path: str, num_frames: int, fps: int) -> None:
        height, width = 32, 32
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("Failed to create step12 test video.")

        for frame_idx in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            if frame_idx % 3 == 0:
                frame[..., 2] = 220
            elif frame_idx % 3 == 1:
                frame[..., 1] = 220
            else:
                frame[..., 0] = 220
            frame[:, frame_idx % width, :] = 255
            writer.write(frame)
        writer.release()

    def test_end_to_end_inference_engine(self):
        engine = VideoSummaryInferenceEngine(config=self.config)

        result = engine.run(
            video_path=self.video_path,
            title="Travel vlog in downtown",
            category="travel",
            asr_segments=[
                {"start_sec": 0.0, "end_sec": 1.2, "text": "We arrive in the city center."},
                {"start_sec": 1.2, "end_sec": 3.0, "text": "The street gets busy as we keep moving."},
            ],
            output_path=self.output_path,
        )

        self.assertIn("summary", result)
        self.assertIn("frame_scores", result)
        self.assertIn("decision_logs", result)
        self.assertIn("selected_segments", result)
        self.assertTrue(os.path.exists(self.output_path))
        self.assertTrue(result["decision_logs"])
        self.assertEqual(len(result["frame_scores"]), result["video_meta"]["total_frames"])

        with open(self.output_path, "r", encoding="utf-8") as file_obj:
            saved = json.load(file_obj)

        self.assertIn("global_context", saved)
        self.assertIn("summary", saved)
        self.assertIn("output_path", result)


if __name__ == "__main__":
    unittest.main()