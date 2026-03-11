from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image

from src.data.schemas import VideoInfo


def load_video_info(video_path: str) -> VideoInfo:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()

    if fps <= 0:
        raise ValueError(f"Invalid fps for video: {video_path}")

    return VideoInfo(
        video_id=Path(video_path).stem,
        video_path=video_path,
        fps=fps,
        total_frames=total_frames,
    )


def read_frames(video_path: str, frame_indices: list[int]) -> list[Image.Image]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    images: list[Image.Image] = []
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = capture.read()
        if not success:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(rgb))

    capture.release()
    return images