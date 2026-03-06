import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.schemas import MemoryState, Segment, VideoMeta, Window, WindowFeature, WindowScore


class TestSchemas(unittest.TestCase):
    def test_video_meta_serialization(self):
        meta = VideoMeta(
            video_id="demo_001",
            file_path="data/raw/demo.mp4",
            fps=25.0,
            total_frames=250,
            duration_sec=10.0,
            title="Demo",
        )
        meta_dict = meta.to_dict()

        self.assertEqual(meta_dict["video_id"], "demo_001")
        self.assertEqual(meta_dict["fps"], 25.0)

    def test_segment_and_window_ranges(self):
        segment = Segment(
            seg_id="seg_0",
            start_frame=0,
            end_frame=120,
            start_sec=0.0,
            end_sec=4.8,
        )
        window = Window(
            win_id="win_0",
            start_frame=0,
            end_frame=32,
            sampled_frame_indices=[0, 8, 16, 24, 32],
        )

        self.assertEqual(segment.num_frames, 120)
        self.assertEqual(window.num_frames, 32)

    def test_window_feature_and_score(self):
        feature = WindowFeature(
            win_id="win_1",
            visual_description="A person walks into a room.",
            local_caption="The subject enters the room.",
            semantic_summary="A room-entry event begins.",
            evidence_notes=["person", "room", "entry"],
        )
        score = WindowScore(
            win_id="win_1",
            expert_opinions={"mainline": "强"},
            base_decision="建议保留",
            cf_comment="删除后会丢失入场信息。",
            final_importance="建议保留",
        )

        self.assertEqual(feature.evidence_notes[0], "person")
        self.assertEqual(score.final_importance, "建议保留")

    def test_memory_state_defaults(self):
        memory = MemoryState()
        self.assertEqual(memory.to_dict(), {
            "story_slots": [],
            "selected_slots": [],
            "redundancy_bank": [],
            "temporal_context": [],
        })

    def test_invalid_range_raises(self):
        with self.assertRaises(ValueError):
            Segment(
                seg_id="seg_invalid",
                start_frame=10,
                end_frame=5,
                start_sec=1.0,
                end_sec=0.5,
            )

    def test_default_config_contains_step1_contracts(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config_text = f.read()

        self.assertIn("project:", config_text)
        self.assertIn("data:", config_text)
        self.assertIn("video:", config_text)
        self.assertIn("window:", config_text)
        self.assertIn("expected_time_units:", config_text)
        self.assertIn("- frame", config_text)
        self.assertIn("- second", config_text)


if __name__ == "__main__":
    unittest.main()