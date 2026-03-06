from __future__ import annotations

from typing import List

from src.scoring.frame_fusion import FrameScoreFusion, PRIORITY_VALUE


class BudgetedSummarySelector:
    """Greedy selector for frame-level labels under a frame budget."""

    def __init__(
        self,
        frame_fusion: FrameScoreFusion | None = None,
        min_label: str = "low",
        allow_partial_segment: bool = True,
    ):
        self.frame_fusion = frame_fusion or FrameScoreFusion()
        self.min_priority = PRIORITY_VALUE.get(min_label, PRIORITY_VALUE["low"])
        self.allow_partial_segment = allow_partial_segment

    def select(self, frame_scores: List[str], budget: int | float) -> List[dict]:
        total_frames = len(frame_scores)
        budget_frames = self._resolve_budget_frames(budget, total_frames)
        if budget_frames <= 0 or total_frames == 0:
            return []

        candidate_segments = [
            segment
            for segment in self.frame_fusion.extract_candidate_segments(frame_scores)
            if segment["priority"] >= self.min_priority
        ]
        ranked_segments = sorted(
            candidate_segments,
            key=lambda item: (-item["priority"], -item["length"], item["start_frame"]),
        )

        selected_segments: List[dict] = []
        used_budget = 0
        for segment in ranked_segments:
            if used_budget >= budget_frames:
                break
            remaining = budget_frames - used_budget
            if segment["length"] > remaining:
                if self.allow_partial_segment and remaining > 0:
                    selected_segments.append(
                        {
                            **segment,
                            "end_frame": segment["start_frame"] + remaining,
                            "length": remaining,
                        }
                    )
                    used_budget += remaining
                continue
            selected_segments.append(segment)
            used_budget += segment["length"]

        selected_segments.sort(key=lambda item: item["start_frame"])
        return selected_segments

    def _resolve_budget_frames(self, budget: int | float, total_frames: int) -> int:
        if isinstance(budget, float):
            if budget < 0:
                raise ValueError(f"budget must be non-negative, got {budget}.")
            if budget <= 1.0:
                return int(total_frames * budget)
            return int(budget)

        if budget < 0:
            raise ValueError(f"budget must be non-negative, got {budget}.")
        return budget