import json
import os
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.schemas import WindowFeature
from src.evaluation import TrainingFreeAblationRunner
from src.memory.memory_bank import MemoryBank


class TestOptimizationAndAblationStep14(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="video_step14_")
        self.video_path = os.path.join(self.temp_dir, "ablation_demo.avi")
        self._create_test_video(self.video_path, num_frames=12, fps=6)

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
            raise RuntimeError("Failed to create step14 test video.")

        for frame_idx in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[..., frame_idx % 3] = 200
            frame[:, frame_idx % width, :] = 255
            writer.write(frame)
        writer.release()

    def test_memory_similarity_prune(self):
        memory_bank = MemoryBank(topk=2, max_items_per_slot=5, similarity_prune_threshold=0.5)
        feature_a = WindowFeature(
            win_id="w0",
            visual_description="Busy downtown street scene.",
            local_caption="Pedestrians move through downtown.",
            semantic_summary="Local event: pedestrians move through a busy downtown street.",
            evidence_notes=[],
        )
        feature_b = WindowFeature(
            win_id="w1",
            visual_description="Crowded downtown street.",
            local_caption="Pedestrians move through the same downtown street.",
            semantic_summary="Local event: pedestrians move through the busy downtown street.",
            evidence_notes=[],
        )

        memory_bank.update(feature_a, final_decision="建议保留", is_selected=True)
        memory_bank.update(feature_b, final_decision="建议保留", is_selected=True)

        snapshot = memory_bank.snapshot()
        self.assertEqual(len(snapshot.story_slots), 1)
        self.assertEqual(len(snapshot.selected_slots), 1)

    def test_ablation_runner_outputs_report(self):
        runner = TrainingFreeAblationRunner(
            base_config={
                "segment": {"coarse_segment_sec": 1.0},
                "window": {"win_len_sec": 1.0, "overlap_sec": 0.5, "sample_rate": 1},
                "summary": {"budget_ratio": 0.5},
                "memory": {"topk": 3, "max_items_per_slot": 8, "similarity_prune_threshold": 0.8},
                "optimization": {"selection": {"min_label": "low", "allow_partial_segment": True}},
            }
        )

        report = runner.run(
            video_path=self.video_path,
            output_dir=os.path.join(self.temp_dir, "ablation"),
            title="Travel walkthrough",
            category="travel",
            asr_segments=[{"start_sec": 0.0, "end_sec": 2.0, "text": "We keep moving along the street."}],
            gt_scores=[0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
            variants={
                "full_system": {},
                "no_memory": {"ablation": {"disable_memory": True}},
            },
        )

        self.assertTrue(os.path.exists(report["report_path"]))
        self.assertIn("full_system", report["variants"])
        self.assertIn("no_memory", report["variants"])
        self.assertIn("metrics", report["variants"]["full_system"])

        no_memory_output = report["variants"]["no_memory"]["output_path"]
        with open(no_memory_output, "r", encoding="utf-8") as file_obj:
            saved = json.load(file_obj)

        self.assertEqual(saved["final_memory"]["story_slots"], [])
        self.assertIn("runtime_profile", saved)


if __name__ == "__main__":
    unittest.main()