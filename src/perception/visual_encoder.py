from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.datasets.schemas import Window


class WindowVisualEncoder:
    """Generate a stable text description from sampled video frames."""

    def __init__(self, resolution: Optional[int] = None):
        self.resolution = resolution

    def extract_sampled_frames(self, video_path: str, window: Window) -> np.ndarray:
        """Load sampled frames for a window from disk as RGB numpy arrays."""
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        frames = []
        try:
            for frame_index in window.sampled_frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if not success:
                    raise ValueError(
                        f"Failed to read frame {frame_index} from video: {video_path}"
                    )

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.resolution is not None:
                    frame = cv2.resize(frame, (self.resolution, self.resolution))
                frames.append(frame)
        finally:
            capture.release()

        if not frames:
            raise ValueError(f"No frames were extracted for window {window.win_id}.")

        return np.stack(frames, axis=0)

    def describe_window(self, window: Window, video_path: Optional[str] = None, frames=None) -> str:
        """Return a structured visual description for the given window."""
        analysis = self.analyze_window(window=window, video_path=video_path, frames=frames)

        return (
            f"Window {window.win_id}: sampled {analysis['num_frames']} frames from {window.start_frame} to "
            f"{window.end_frame}; brightness={analysis['brightness']}; "
            f"dominant_color={analysis['dominant_color']}; motion={analysis['motion']}; "
            f"scene_variation={analysis['scene_variation']}."
        )

    def analyze_window(self, window: Window, video_path: Optional[str] = None, frames=None) -> dict:
        """Return a lightweight analysis dictionary for the given window."""
        if frames is None:
            if video_path is None:
                raise ValueError("Either video_path or frames must be provided.")
            frames_np = self.extract_sampled_frames(video_path, window)
        else:
            frames_np = self.prepare_frames(frames)

        return {
            "win_id": window.win_id,
            "num_frames": int(frames_np.shape[0]),
            "brightness": self._estimate_brightness(frames_np),
            "dominant_color": self._estimate_dominant_color(frames_np),
            "motion": self._estimate_motion(frames_np),
            "scene_variation": self._estimate_scene_variation(frames_np),
        }

    def prepare_frames(self, frames) -> np.ndarray:
        if hasattr(frames, "detach"):
            frames = frames.detach().cpu().numpy()

        frames_np = np.asarray(frames)
        if frames_np.ndim != 4:
            raise ValueError(
                f"Expected frames with 4 dimensions [T, C, H, W] or [T, H, W, C], got {frames_np.shape}."
            )

        if frames_np.shape[1] in (1, 3):
            frames_np = np.transpose(frames_np, (0, 2, 3, 1))

        if frames_np.shape[-1] != 3:
            raise ValueError(f"Expected RGB frames with 3 channels, got {frames_np.shape}.")

        if frames_np.dtype != np.uint8:
            max_value = float(frames_np.max()) if frames_np.size else 0.0
            if max_value <= 1.0:
                frames_np = (frames_np * 255.0).clip(0, 255).astype(np.uint8)
            else:
                frames_np = frames_np.clip(0, 255).astype(np.uint8)

        if self.resolution is not None:
            resized_frames = [
                cv2.resize(frame, (self.resolution, self.resolution)) for frame in frames_np
            ]
            frames_np = np.stack(resized_frames, axis=0)

        return frames_np

    def _estimate_brightness(self, frames: np.ndarray) -> str:
        gray = frames.mean(axis=-1)
        score = float(gray.mean() / 255.0)
        if score < 0.25:
            return "dark"
        if score < 0.65:
            return "balanced"
        return "bright"

    def _estimate_dominant_color(self, frames: np.ndarray) -> str:
        channel_means = frames.mean(axis=(0, 1, 2))
        channel_names = ["red", "green", "blue"]
        dominant_index = int(np.argmax(channel_means))
        sorted_values = np.sort(channel_means)

        if float(sorted_values[-1] - sorted_values[-2]) < 12.0:
            return "balanced"
        return channel_names[dominant_index]

    def _estimate_motion(self, frames: np.ndarray) -> str:
        if frames.shape[0] <= 1:
            return "static"

        gray = frames.mean(axis=-1).astype(np.float32)
        diffs = np.abs(np.diff(gray, axis=0)) / 255.0
        score = float(diffs.mean())

        if score < 0.03:
            return "static"
        if score < 0.12:
            return "moderate"
        return "dynamic"

    def _estimate_scene_variation(self, frames: np.ndarray) -> str:
        if frames.shape[0] <= 1:
            return "stable"

        frame_means = frames.mean(axis=(1, 2, 3)).astype(np.float32) / 255.0
        score = float(frame_means.std())
        if score < 0.04:
            return "stable"
        if score < 0.12:
            return "changing"
        return "highly changing"