from __future__ import annotations

from typing import Dict, List

from src.datasets.schemas import Window, WindowScore


PRIORITY_MAP: Dict[str, str] = {
    "必须保留": "high",
    "建议保留": "medium",
    "可选": "low",
    "建议省略": "drop",
}

PRIORITY_VALUE: Dict[str, int] = {
    "drop": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}


class FrameScoreFusion:
    """Fuse window-level importance decisions into frame-level labels."""

    def fuse(
        self,
        window_scores: List[WindowScore],
        windows: List[Window],
        total_frames: int,
    ) -> List[str]:
        if total_frames < 0:
            raise ValueError(f"total_frames must be non-negative, got {total_frames}.")
        if total_frames == 0:
            return []

        frame_values = [PRIORITY_VALUE["drop"] for _ in range(total_frames)]
        window_lookup = {window.win_id: window for window in windows}

        for window_score in window_scores:
            window = window_lookup.get(window_score.win_id)
            if window is None:
                continue

            label = PRIORITY_MAP.get(window_score.final_importance, "drop")
            value = PRIORITY_VALUE[label]
            start = max(0, window.start_frame)
            end = min(total_frames, window.end_frame)

            for frame_index in range(start, end):
                frame_values[frame_index] = max(frame_values[frame_index], value)

        return [self._value_to_label(value) for value in frame_values]

    def extract_candidate_segments(self, frame_scores: List[str]) -> List[dict]:
        if not frame_scores:
            return []

        segments: List[dict] = []
        start = None
        current_label = None

        for index, label in enumerate(frame_scores):
            if label == "drop":
                if current_label is not None and start is not None:
                    segments.append(self._build_segment(start, index, current_label))
                    start = None
                    current_label = None
                continue

            if current_label is None:
                start = index
                current_label = label
                continue

            if label != current_label:
                segments.append(self._build_segment(start, index, current_label))
                start = index
                current_label = label

        if current_label is not None and start is not None:
            segments.append(self._build_segment(start, len(frame_scores), current_label))

        return segments

    def _build_segment(self, start_frame: int, end_frame: int, label: str) -> dict:
        return {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "label": label,
            "length": end_frame - start_frame,
            "priority": PRIORITY_VALUE[label],
        }

    def _value_to_label(self, value: int) -> str:
        for label, numeric in PRIORITY_VALUE.items():
            if numeric == value:
                return label
        return "drop"