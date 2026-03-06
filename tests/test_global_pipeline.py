import os
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.schemas import VideoMeta
from src.pipeline.global_pipeline import GlobalUnderstandingPipeline
from src.preprocessing.segmenter import build_fixed_segments
from src.preprocessing.video_loader import load_video_meta


class TestGlobalUnderstandingStep6(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="video_step6_")
        self.video_path = os.path.join(self.temp_dir, "global_demo.avi")
        self._create_test_video(self.video_path, num_frames=12, fps=4)
        self.pipeline = GlobalUnderstandingPipeline(sample_rate=2)

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
            raise RuntimeError("Failed to create global pipeline test video.")

        for frame_idx in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            if frame_idx < num_frames // 2:
                frame[..., 2] = 180
            else:
                frame[..., 1] = 180
            writer.write(frame)
        writer.release()

    def test_build_global_context(self):
        meta = load_video_meta(
            self.video_path,
            title="Travel vlog in the city",
            asr_segments=[
                {"start_sec": 0.0, "end_sec": 1.5, "text": "We arrive downtown"},
                {"start_sec": 1.5, "end_sec": 3.0, "text": "The street gets busy"},
            ],
            category="travel",
        )
        segments = build_fixed_segments(meta, segment_length_sec=1.5)

        result = self.pipeline.build_global_context(video_meta=meta, segments=segments)

        self.assertEqual(len(result["global_captions"]), 2)
        self.assertIn("Segment 1:", result["global_story"])
        self.assertIn("Primary theme: travel.", result["summary_goal"])
        self.assertIn("travel", result["theme_dist"])
        self.assertAlmostEqual(sum(result["theme_dist"].values()), 1.0, places=3)

    def test_build_global_context_without_segments(self):
        meta = VideoMeta(
            video_id="empty_demo",
            file_path=self.video_path,
            fps=4.0,
            total_frames=0,
            duration_sec=0.0,
            title="Empty clip",
        )

        result = self.pipeline.build_global_context(video_meta=meta, segments=[])

        self.assertEqual(result["global_captions"], [])
        self.assertIn("No coarse segments available", result["global_story"])
        self.assertIn("Primary theme:", result["summary_goal"])


if __name__ == "__main__":
    unittest.main()