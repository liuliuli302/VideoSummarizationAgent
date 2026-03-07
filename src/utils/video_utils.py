"""Video I/O helpers used by baseline training and debugging scripts."""

from __future__ import annotations

import os
from typing import Iterable

import cv2
import numpy as np
import torch

try:
    import decord
except ImportError:  # pragma: no cover - optional dependency path
    decord = None


def load_video_frames(video_path: str, num_frames: int = 16, resolution: int = 224) -> torch.Tensor:
    """Load uniformly sampled frames from a video.

    The loader prefers `decord` for speed, but falls back to OpenCV when decord
    is unavailable. A zero tensor is returned only when decoding fully fails.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}.")
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        if decord is not None:
            return _load_with_decord(video_path=video_path, num_frames=num_frames, resolution=resolution)
        return _load_with_cv2(video_path=video_path, num_frames=num_frames, resolution=resolution)
    except Exception as exc:
        print(f"Error loading video {video_path}: {exc}")
        return torch.zeros(num_frames, 3, resolution, resolution, dtype=torch.float32)


def save_video_tensor(tensor: torch.Tensor | np.ndarray, output_path: str, fps: int = 8) -> str:
    """Persist a tensor video clip to disk.

    Args:
        tensor: Video tensor shaped as `[T, C, H, W]` or `[T, H, W, C]`.
        output_path: Destination file path.
        fps: Frames per second written to the output video.

    Returns:
        The output path for convenient chaining in experiments.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")

    frames = _normalize_video_tensor(tensor)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    height, width = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create output video: {output_path}")

    try:
        for frame in frames:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)
    finally:
        writer.release()

    return output_path


def _load_with_decord(video_path: str, num_frames: int, resolution: int) -> torch.Tensor:
    video_reader = decord.VideoReader(video_path, width=resolution, height=resolution)
    total_frames = len(video_reader)
    indices = _uniform_indices(total_frames=total_frames, num_frames=num_frames)
    frames = video_reader.get_batch(indices).asnumpy()
    return _to_tensor(frames, num_frames=num_frames)


def _load_with_cv2(video_path: str, num_frames: int, resolution: int) -> torch.Tensor:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = _uniform_indices(total_frames=total_frames, num_frames=num_frames)
    frames = []
    try:
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if not success:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (resolution, resolution))
            frames.append(frame)
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"No frames decoded from video: {video_path}")
    return _to_tensor(np.stack(frames, axis=0), num_frames=num_frames)


def _uniform_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.zeros((1,), dtype=int)
    if total_frames <= num_frames:
        return np.arange(total_frames, dtype=int)
    return np.linspace(0, total_frames - 1, num_frames, dtype=int)


def _to_tensor(frames: np.ndarray, num_frames: int) -> torch.Tensor:
    frames_tensor = torch.from_numpy(frames).float() / 255.0
    frames_tensor = frames_tensor.permute(0, 3, 1, 2)
    if frames_tensor.shape[0] < num_frames:
        pad_len = num_frames - frames_tensor.shape[0]
        padding = frames_tensor[-1:].repeat(pad_len, 1, 1, 1)
        frames_tensor = torch.cat([frames_tensor, padding], dim=0)
    return frames_tensor[:num_frames]


def _normalize_video_tensor(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = np.asarray(tensor)

    if array.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape {array.shape}.")

    if array.shape[1] in (1, 3):
        array = np.transpose(array, (0, 2, 3, 1))
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB video tensor, got shape {array.shape}.")

    if array.dtype != np.uint8:
        max_value = float(array.max()) if array.size else 0.0
        if max_value <= 1.0:
            array = (array * 255.0).clip(0, 255).astype(np.uint8)
        else:
            array = array.clip(0, 255).astype(np.uint8)
    return array
