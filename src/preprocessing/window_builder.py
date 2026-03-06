from __future__ import annotations

from typing import List

from src.datasets.schemas import VideoMeta, Window


def uniform_sample_indices(start_frame: int, end_frame: int, sample_rate: int) -> List[int]:
    """Generate frame indices inside ``[start_frame, end_frame)`` with fixed stride."""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
    if end_frame < start_frame:
        raise ValueError(
            f"end_frame must be greater than or equal to start_frame, got {start_frame} -> {end_frame}."
        )
    if end_frame == start_frame:
        return [start_frame]

    indices = list(range(start_frame, end_frame, sample_rate))
    last_valid_frame = end_frame - 1
    if not indices:
        return [start_frame]
    if indices[-1] != last_valid_frame:
        indices.append(last_valid_frame)
    return indices


def build_sliding_windows(
    video_meta: VideoMeta,
    win_len_sec: float,
    overlap_sec: float,
    sample_rate: int,
) -> List[Window]:
    """Build fixed-length sliding windows from a video metadata object."""
    if win_len_sec <= 0:
        raise ValueError(f"win_len_sec must be positive, got {win_len_sec}.")
    if overlap_sec < 0:
        raise ValueError(f"overlap_sec must be non-negative, got {overlap_sec}.")
    if video_meta.total_frames == 0:
        return []

    win_len_frames = max(1, int(round(win_len_sec * video_meta.fps)))
    overlap_frames = int(round(overlap_sec * video_meta.fps))
    step = win_len_frames - overlap_frames

    if step <= 0:
        raise ValueError(
            "overlap_sec is too large. Computed sliding step must be positive, "
            f"got step={step}."
        )

    windows: List[Window] = []
    start_frame = 0
    win_index = 0

    while start_frame < video_meta.total_frames:
        end_frame = min(start_frame + win_len_frames, video_meta.total_frames)
        sampled_frame_indices = uniform_sample_indices(start_frame, end_frame, sample_rate)

        windows.append(
            Window(
                win_id=f"w_{win_index}",
                start_frame=start_frame,
                end_frame=end_frame,
                sampled_frame_indices=sampled_frame_indices,
            )
        )

        if end_frame == video_meta.total_frames:
            break

        start_frame += step
        win_index += 1

    return windows