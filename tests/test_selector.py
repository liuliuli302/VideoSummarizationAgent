import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.schemas import Window, WindowScore
from src.scoring.frame_fusion import FrameScoreFusion
from src.scoring.selector import BudgetedSummarySelector


class TestSelectionStep11(unittest.TestCase):
    def setUp(self):
        self.windows = [
            Window(win_id="w_0", start_frame=0, end_frame=4, sampled_frame_indices=[0, 2, 3]),
            Window(win_id="w_1", start_frame=2, end_frame=6, sampled_frame_indices=[2, 4, 5]),
            Window(win_id="w_2", start_frame=6, end_frame=10, sampled_frame_indices=[6, 8, 9]),
        ]
        self.window_scores = [
            WindowScore(
                win_id="w_0",
                expert_opinions={},
                base_decision="Final Importance: 建议保留\nReason: test\nConflict Handling: test",
                cf_comment="未启用 CMCC",
                final_importance="建议保留",
            ),
            WindowScore(
                win_id="w_1",
                expert_opinions={},
                base_decision="Final Importance: 必须保留\nReason: test\nConflict Handling: test",
                cf_comment="未启用 CMCC",
                final_importance="必须保留",
            ),
            WindowScore(
                win_id="w_2",
                expert_opinions={},
                base_decision="Final Importance: 可选\nReason: test\nConflict Handling: test",
                cf_comment="未启用 CMCC",
                final_importance="可选",
            ),
        ]
        self.fusion = FrameScoreFusion()
        self.selector = BudgetedSummarySelector(frame_fusion=self.fusion)

    def test_frame_score_fusion(self):
        frame_scores = self.fusion.fuse(
            window_scores=self.window_scores,
            windows=self.windows,
            total_frames=10,
        )

        self.assertEqual(len(frame_scores), 10)
        self.assertEqual(frame_scores[0], "medium")
        self.assertEqual(frame_scores[2], "high")
        self.assertEqual(frame_scores[8], "low")

    def test_extract_candidate_segments(self):
        frame_scores = ["medium", "medium", "high", "high", "high", "high", "low", "low", "low", "low"]
        segments = self.fusion.extract_candidate_segments(frame_scores)

        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0]["label"], "medium")
        self.assertEqual(segments[1]["label"], "high")
        self.assertEqual(segments[2]["start_frame"], 6)

    def test_budgeted_selection(self):
        frame_scores = self.fusion.fuse(
            window_scores=self.window_scores,
            windows=self.windows,
            total_frames=10,
        )
        selected = self.selector.select(frame_scores=frame_scores, budget=0.6)

        self.assertTrue(selected)
        self.assertIn("high", [segment["label"] for segment in selected])
        total_selected_frames = sum(segment["length"] for segment in selected)
        self.assertLessEqual(total_selected_frames, 6)


if __name__ == "__main__":
    unittest.main()