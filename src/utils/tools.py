"""Small helper utilities used by the baseline `VideoAgent` implementation."""

from __future__ import annotations

import hashlib

import numpy as np
import torch


class VideoTools:
    """Utility collection for simple tensor-side operations.

    The project uses lightweight deterministic helpers rather than external
    detection stacks so that unit tests stay fast and reproducible.
    """

    def extract_clip(self, video_tensor: torch.Tensor, start_frame: int, end_frame: int) -> torch.Tensor:
        """Extract a temporal clip from a `[T, C, H, W]` video tensor."""
        if video_tensor.ndim != 4:
            raise ValueError(f"video_tensor must have shape [T, C, H, W], got {tuple(video_tensor.shape)}")

        total_frames = video_tensor.shape[0]
        start = max(0, int(start_frame))
        end = min(total_frames, int(end_frame))
        if end <= start:
            return video_tensor[start:start]
        return video_tensor[start:end]

    def detect_object(self, frame_tensor: torch.Tensor, object_name: str) -> float:
        """Return a deterministic pseudo confidence for debugging and demos.

        This is not a real detector. It intentionally avoids randomness so the
        same input frame always yields the same score.
        """
        frame_array = self._to_numpy(frame_tensor)
        if frame_array.size == 0:
            return 0.0

        frame_score = float(frame_array.mean()) / 255.0
        token_score = self._stable_text_score(object_name)
        return float(np.clip((0.7 * frame_score) + (0.3 * token_score), 0.0, 1.0))

    def _to_numpy(self, frame_tensor: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(frame_tensor, torch.Tensor):
            array = frame_tensor.detach().cpu().numpy()
        else:
            array = np.asarray(frame_tensor)

        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            max_value = float(array.max()) if array.size else 0.0
            if max_value <= 1.0:
                array = (array * 255.0).clip(0, 255).astype(np.uint8)
            else:
                array = array.clip(0, 255).astype(np.uint8)
        return array

    def _stable_text_score(self, text: str) -> float:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="big") / float(2**32 - 1)
