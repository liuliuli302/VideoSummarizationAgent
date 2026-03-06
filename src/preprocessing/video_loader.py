from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from src.datasets.schemas import VideoMeta


def load_video_meta(
    video_path: str,
    video_id: Optional[str] = None,
    title: Optional[str] = None,
    asr_segments: Optional[List[Dict[str, Any]]] = None,
    category: Optional[str] = None,
) -> VideoMeta:
    """Read basic video metadata and convert it into a ``VideoMeta`` object."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()

    if fps <= 0:
        raise ValueError(f"Invalid fps read from video: {video_path}")
    if total_frames < 0:
        raise ValueError(f"Invalid frame count read from video: {video_path}")

    resolved_video_id = video_id or Path(video_path).stem
    duration_sec = total_frames / fps if total_frames > 0 else 0.0

    return VideoMeta(
        video_id=resolved_video_id,
        file_path=video_path,
        fps=fps,
        total_frames=total_frames,
        duration_sec=duration_sec,
        title=title,
        asr_segments=asr_segments,
        category=category,
    )