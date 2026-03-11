from __future__ import annotations


def build_segments_by_count(
    total_frames: int,
    num_segments: int,
    caption_frames_per_segment: int,
) -> list[dict[str, int | list[int]]]:
    if total_frames <= 0:
        return []
    if num_segments <= 0:
        raise ValueError("num_segments must be positive.")

    segment_size = max(1, total_frames // num_segments)
    segments = []
    start_frame = 0

    for segment_id in range(num_segments):
        if start_frame >= total_frames:
            break
        end_frame = total_frames if segment_id == num_segments - 1 else min(total_frames, start_frame + segment_size)
        segments.append(
            {
                "segment_id": segment_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "caption_frame_indices": sample_uniform_indices(start_frame, end_frame, caption_frames_per_segment),
            }
        )
        start_frame = end_frame

    return segments


def build_segments_by_frame_window(
    total_frames: int,
    frames_per_segment: int,
    caption_frames_per_segment: int,
    overlap_frames: int = 0,
) -> list[dict[str, int | list[int]]]:
    if total_frames <= 0:
        return []
    if frames_per_segment <= 0:
        raise ValueError("frames_per_segment must be positive.")
    if overlap_frames < 0:
        raise ValueError("overlap_frames must be non-negative.")
    if overlap_frames >= frames_per_segment:
        raise ValueError("overlap_frames must be smaller than frames_per_segment.")

    segments = []
    segment_id = 0
    stride = frames_per_segment - overlap_frames
    for start_frame in range(0, total_frames, stride):
        end_frame = min(total_frames, start_frame + frames_per_segment)
        segments.append(
            {
                "segment_id": segment_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "caption_frame_indices": sample_uniform_indices(start_frame, end_frame, caption_frames_per_segment),
            }
        )
        if end_frame >= total_frames:
            break
        segment_id += 1

    return segments


def sample_uniform_indices(start_frame: int, end_frame: int, num_samples: int) -> list[int]:
    if end_frame <= start_frame or num_samples <= 0:
        return []

    length = end_frame - start_frame
    count = min(num_samples, length)
    if count == 1:
        return [start_frame]

    return [
        int(round(start_frame + index * (length - 1) / (count - 1)))
        for index in range(count)
    ]