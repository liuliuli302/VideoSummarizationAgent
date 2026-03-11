from __future__ import annotations

from src.data.schemas import SegmentScore


class FrameScoreMapper:
    def assign_segment_scores_to_original_frames(
        self,
        segment_scores: list[SegmentScore],
        original_picks: list[int],
        total_frames: int,
    ) -> list[float]:
        if total_frames <= 0:
            return []

        dense_score_sums = [0.0] * total_frames
        dense_score_counts = [0] * total_frames
        for segment_score in segment_scores:
            start = max(0, int(segment_score.start_frame))
            end = min(total_frames, int(segment_score.end_frame))
            for frame_index in range(start, end):
                dense_score_sums[frame_index] += float(segment_score.final_score)
                dense_score_counts[frame_index] += 1

        dense_scores = [
            (dense_score_sums[index] / dense_score_counts[index]) if dense_score_counts[index] > 0 else 0.0
            for index in range(total_frames)
        ]

        return [dense_scores[pick] if 0 <= pick < total_frames else 0.0 for pick in original_picks]