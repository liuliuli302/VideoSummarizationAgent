import os
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.segmenter import build_fixed_segments
from src.preprocessing.video_loader import load_video_meta
from src.preprocessing.window_builder import build_sliding_windows, uniform_sample_indices


class TestPreprocessingStep2(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="video_step2_")
        self.video_path = os.path.join(self.temp_dir, "demo.avi")
        self._create_test_video(self.video_path, num_frames=10, fps=5)

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
            raise RuntimeError("Failed to create test video.")

        for frame_idx in range(num_frames):
            pixel_value = np.uint8((frame_idx * 20) % 255)
            frame = np.full((height, width, 3), pixel_value, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def test_load_video_meta(self):
        meta = load_video_meta(self.video_path)

        self.assertEqual(meta.video_id, "demo")
        self.assertEqual(meta.total_frames, 10)
        self.assertAlmostEqual(meta.fps, 5.0, places=2)
        self.assertAlmostEqual(meta.duration_sec, 2.0, places=2)

    def test_build_fixed_segments(self):
        meta = load_video_meta(self.video_path)
        segments = build_fixed_segments(meta, segment_length_sec=1.0)

        self.assertEqual(len(segments), 2)
        self.assertEqual((segments[0].start_frame, segments[0].end_frame), (0, 5))
        self.assertEqual((segments[1].start_frame, segments[1].end_frame), (5, 10))
        self.assertAlmostEqual(segments[1].end_sec, 2.0, places=2)

    def test_uniform_sample_indices(self):
        indices = uniform_sample_indices(0, 5, sample_rate=2)
        self.assertEqual(indices, [0, 2, 4])

        last_short = uniform_sample_indices(6, 10, sample_rate=3)
        self.assertEqual(last_short, [6, 9])

    def test_build_sliding_windows(self):
        meta = load_video_meta(self.video_path)
        windows = build_sliding_windows(
            meta,
            win_len_sec=0.8,
            overlap_sec=0.4,
            sample_rate=2,
        )

        self.assertEqual(len(windows), 4)
        self.assertEqual((windows[0].start_frame, windows[0].end_frame), (0, 4))
        self.assertEqual((windows[1].start_frame, windows[1].end_frame), (2, 6))
        self.assertEqual((windows[-1].start_frame, windows[-1].end_frame), (6, 10))
        self.assertEqual(windows[0].sampled_frame_indices, [0, 2, 3])
        self.assertEqual(windows[-1].sampled_frame_indices, [6, 8, 9])

    def test_invalid_overlap_raises(self):
        meta = load_video_meta(self.video_path)
        with self.assertRaises(ValueError):
            build_sliding_windows(meta, win_len_sec=0.4, overlap_sec=0.4, sample_rate=1)


if __name__ == "__main__":
    unittest.main()