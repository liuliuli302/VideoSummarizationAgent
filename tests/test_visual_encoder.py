import os
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.schemas import Window
from src.perception.visual_encoder import WindowVisualEncoder


class TestVisualEncoderStep3(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="video_step3_")
        self.video_path = os.path.join(self.temp_dir, "visual_demo.avi")
        self._create_motion_video(self.video_path)
        self.window = Window(
            win_id="w_0",
            start_frame=0,
            end_frame=8,
            sampled_frame_indices=[0, 2, 4, 7],
        )
        self.encoder = WindowVisualEncoder(resolution=32)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_motion_video(self, video_path: str) -> None:
        height, width = 32, 32
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            4,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("Failed to create test video for visual encoder.")

        for frame_idx in range(8):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            left = min(frame_idx * 3, width - 8)
            frame[8:24, left:left + 8, 2] = 255
            writer.write(frame)

        writer.release()

    def test_extract_sampled_frames(self):
        frames = self.encoder.extract_sampled_frames(self.video_path, self.window)
        self.assertEqual(frames.shape, (4, 32, 32, 3))
        self.assertGreater(float(frames[..., 0].mean()), 10.0)

    def test_describe_window_from_video(self):
        description = self.encoder.describe_window(window=self.window, video_path=self.video_path)

        self.assertIn("Window w_0", description)
        self.assertIn("sampled 4 frames", description)
        self.assertIn("dominant_color=red", description)
        self.assertIn("motion=", description)
        self.assertIn("scene_variation=", description)

    def test_describe_window_from_numpy_frames(self):
        frames = np.zeros((4, 16, 16, 3), dtype=np.uint8)
        frames[..., 1] = 220
        description = self.encoder.describe_window(window=self.window, frames=frames)

        self.assertIn("dominant_color=green", description)
        self.assertIn("brightness=balanced", description)
        self.assertIn("motion=static", description)


if __name__ == "__main__":
    unittest.main()