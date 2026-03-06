import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.schemas import Window
from src.perception.consistency import ConsistencyChecker
from src.perception.fusion import WindowFeatureBuilder


class TestFusionStep5(unittest.TestCase):
    def setUp(self):
        self.window = Window(
            win_id="w_2",
            start_frame=10,
            end_frame=20,
            sampled_frame_indices=[10, 14, 19],
        )
        self.checker = ConsistencyChecker()
        self.builder = WindowFeatureBuilder(consistency_checker=self.checker)

    def test_consistency_checker_high_alignment(self):
        comment = self.checker.check(
            visual_description="Window w_2 shows a bright red scene with moderate motion.",
            local_caption="A bright red scene with moderate motion and noticeable activity.",
        )
        self.assertIn("Consistency: high", comment)

    def test_consistency_checker_low_alignment(self):
        comment = self.checker.check(
            visual_description="Window w_2 shows a dark blue static scene.",
            local_caption="A bright green environment with strong motion.",
        )
        self.assertIn("Consistency: low", comment)

    def test_build_window_feature(self):
        feature = self.builder.build(
            window=self.window,
            visual_description="Window w_2: sampled 3 frames from 10 to 20; brightness=bright; dominant_color=red; motion=moderate; scene_variation=changing.",
            local_caption="A bright red-dominant scene with moderate motion and noticeable visual change.",
            semantic_summary="Local event: A bright red-dominant scene with moderate motion and noticeable visual change.",
            asr_text=[{"text": "The crowd is moving forward"}],
        )

        self.assertEqual(feature.win_id, "w_2")
        self.assertIn("Visual Description:", feature.semantic_summary)
        self.assertIn("Local Caption:", feature.semantic_summary)
        self.assertIn("ASR: The crowd is moving forward", feature.semantic_summary)
        self.assertTrue(any(note.startswith("consistency_level=") for note in feature.evidence_notes))
        self.assertTrue(any(note.startswith("asr_present=") for note in feature.evidence_notes))


if __name__ == "__main__":
    unittest.main()