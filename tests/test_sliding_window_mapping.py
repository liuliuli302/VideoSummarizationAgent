import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data import ExpertResult, SegmentScore
from src.preprocessing import FrameScoreMapper, build_segments_by_frame_window


class TestSlidingWindowMapping(unittest.TestCase):
    def test_build_segments_with_overlap(self):
        segments = build_segments_by_frame_window(
            total_frames=10,
            frames_per_segment=4,
            caption_frames_per_segment=2,
            overlap_frames=2,
        )

        self.assertEqual(len(segments), 4)
        self.assertEqual((segments[0]["start_frame"], segments[0]["end_frame"]), (0, 4))
        self.assertEqual((segments[1]["start_frame"], segments[1]["end_frame"]), (2, 6))
        self.assertEqual((segments[2]["start_frame"], segments[2]["end_frame"]), (4, 8))
        self.assertEqual((segments[3]["start_frame"], segments[3]["end_frame"]), (6, 10))

    def test_overlap_region_uses_average_score(self):
        mapper = FrameScoreMapper()
        scores = mapper.assign_segment_scores_to_original_frames(
            segment_scores=[
                SegmentScore(
                    segment_id=0,
                    start_frame=0,
                    end_frame=4,
                    planner_score=0.2,
                    planner_reason="first",
                    expert_results={
                        "story_agent": ExpertResult(agent_name="story_agent", score=0.2, reason="first")
                    },
                    final_score=0.2,
                ),
                SegmentScore(
                    segment_id=1,
                    start_frame=2,
                    end_frame=6,
                    planner_score=0.8,
                    planner_reason="second",
                    expert_results={
                        "story_agent": ExpertResult(agent_name="story_agent", score=0.8, reason="second")
                    },
                    final_score=0.8,
                ),
            ],
            original_picks=[0, 1, 2, 3, 4, 5],
            total_frames=6,
        )

        self.assertEqual(scores, [0.2, 0.2, 0.5, 0.5, 0.8, 0.8])

    def test_invalid_overlap_raises(self):
        with self.assertRaises(ValueError):
            build_segments_by_frame_window(
                total_frames=10,
                frames_per_segment=4,
                caption_frames_per_segment=2,
                overlap_frames=4,
            )


if __name__ == "__main__":
    unittest.main()