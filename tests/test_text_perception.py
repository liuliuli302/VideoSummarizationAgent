import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.schemas import Window
from src.perception.captioner import RuleBasedCaptioner
from src.perception.text_encoder import WindowTextEncoder
from src.perception.visual_encoder import WindowVisualEncoder


class TestTextPerceptionStep4(unittest.TestCase):
    def setUp(self):
        self.window = Window(
            win_id="w_1",
            start_frame=0,
            end_frame=6,
            sampled_frame_indices=[0, 2, 5],
        )
        self.visual_encoder = WindowVisualEncoder()
        self.captioner = RuleBasedCaptioner(visual_encoder=self.visual_encoder)
        self.text_encoder = WindowTextEncoder()

    def test_generate_local_caption_from_frames(self):
        frames = np.zeros((3, 16, 16, 3), dtype=np.uint8)
        frames[..., 2] = 180
        frames[1:, 4:10, 4:10, 2] = 255

        caption = self.captioner.generate_caption(window=self.window, frames=frames)

        self.assertIn("scene", caption)
        self.assertIn("blue-dominant", caption)
        self.assertIn("motion", caption)

    def test_build_semantic_summary_with_title_and_asr(self):
        local_caption = "A well-lit blue-dominant scene with moderate motion and noticeable visual change."
        summary = self.text_encoder.build_semantic_summary(
            local_caption=local_caption,
            title="Travel vlog in the city",
            asr_text=[{"text": "We just arrived downtown"}, {"text": "The street is crowded"}],
        )

        self.assertIn("Local event:", summary)
        self.assertIn("Title context: Travel vlog in the city.", summary)
        self.assertIn("ASR cue: We just arrived downtown. The street is crowded.", summary)

    def test_build_semantic_summary_requires_local_caption(self):
        with self.assertRaises(ValueError):
            self.text_encoder.build_semantic_summary(local_caption="   ")


if __name__ == "__main__":
    unittest.main()