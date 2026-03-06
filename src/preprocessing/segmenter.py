from __future__ import annotations

from typing import List

from src.datasets.schemas import Segment, VideoMeta


def build_fixed_segments(video_meta: VideoMeta, segment_length_sec: float) -> List[Segment]:
    """Split a video into fixed-length coarse segments."""
    if segment_length_sec <= 0:
        raise ValueError(f"segment_length_sec must be positive, got {segment_length_sec}.")

    if video_meta.total_frames == 0:
        return []

    segment_length_frames = max(1, int(round(segment_length_sec * video_meta.fps)))
    segments: List[Segment] = []
    seg_index = 0
    start_frame = 0

    while start_frame < video_meta.total_frames:
        end_frame = min(start_frame + segment_length_frames, video_meta.total_frames)
        start_sec = start_frame / video_meta.fps
        end_sec = end_frame / video_meta.fps

        segments.append(
            Segment(
                seg_id=f"seg_{seg_index}",
                start_frame=start_frame,
                end_frame=end_frame,
                start_sec=start_sec,
                end_sec=end_sec,
            )
        )

        start_frame = end_frame
        seg_index += 1

    return segments