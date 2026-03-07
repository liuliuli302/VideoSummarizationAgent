"""Dataset helpers for the legacy frame-level baseline experiments."""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.video_utils import load_video_frames


SUPPORTED_VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")


class VideoDataset(Dataset):
    """Video dataset with optional CSV/JSON metadata support.

    The baseline training code expects a frame tensor and a frame-level label.
    When only video-level labels are available, labels are expanded to match the
    sampled frame count. When no labels are provided, a deterministic synthetic
    label is generated so dry-runs remain reproducible.
    """

    def __init__(
        self,
        data_root: str,
        metadata_file: Optional[str] = None,
        transform=None,
        num_frames: int = 16,
    ) -> None:
        self.data_root = data_root
        self.transform = transform
        self.num_frames = int(num_frames)
        self.video_files: List[str] = []
        self.labels: Dict[str, Any] = {}

        if self.num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {self.num_frames}.")
        if not os.path.isdir(self.data_root):
            raise FileNotFoundError(f"Dataset root not found: {self.data_root}")

        if metadata_file and os.path.exists(metadata_file):
            print(f"Loading metadata from {metadata_file}")
            self._load_metadata(metadata_file)

        if not self.video_files:
            self.video_files = self._scan_videos(self.data_root)

        print(f"Found {len(self.video_files)} videos in {self.data_root}")

    def __len__(self) -> int:
        return len(self.video_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_name = self.video_files[idx]
        video_path = self._resolve_video_path(video_name)

        try:
            video_tensor = load_video_frames(video_path, num_frames=self.num_frames)
        except Exception as exc:
            print(f"Error loading {video_path}: {exc}")
            video_tensor = torch.zeros(self.num_frames, 3, 224, 224, dtype=torch.float32)

        label_tensor = self._build_label_tensor(self.labels.get(video_name))

        if self.transform:
            video_tensor = self.transform(video_tensor)

        return {"video": video_tensor, "label": label_tensor, "name": video_name}

    def _load_metadata(self, metadata_file: str) -> None:
        file_ext = os.path.splitext(metadata_file)[1].lower()
        if file_ext == ".csv":
            metadata_rows = pd.read_csv(metadata_file).to_dict(orient="records")
        elif file_ext == ".json":
            with open(metadata_file, "r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            if isinstance(payload, dict):
                metadata_rows = payload.get("videos") or payload.get("items") or [payload]
            elif isinstance(payload, list):
                metadata_rows = payload
            else:
                raise ValueError(f"Unsupported JSON metadata format in {metadata_file}")
        else:
            raise ValueError(f"Unsupported metadata format: {metadata_file}")

        for row in metadata_rows:
            if not isinstance(row, dict):
                continue
            video_name = self._extract_video_name(row)
            if not video_name:
                continue
            self.video_files.append(video_name)
            if "label" in row:
                self.labels[video_name] = row["label"]

        self.video_files = sorted(dict.fromkeys(self.video_files))

    def _extract_video_name(self, row: Dict[str, Any]) -> Optional[str]:
        return row.get("video_path") or row.get("video_name") or row.get("id") or row.get("video_id")

    def _scan_videos(self, data_root: str) -> List[str]:
        print(f"Scanning directory {data_root} for videos...")
        matches: List[str] = []
        for suffix in SUPPORTED_VIDEO_SUFFIXES:
            matches.extend(glob.glob(os.path.join(data_root, f"**/*{suffix}"), recursive=True))
        return sorted(os.path.relpath(path, data_root) for path in matches)

    def _resolve_video_path(self, video_name: str) -> str:
        if os.path.isabs(video_name):
            return video_name
        return os.path.join(self.data_root, video_name)

    def _build_label_tensor(self, label_value: Any) -> torch.Tensor:
        if label_value is None:
            return self._build_default_label()

        if isinstance(label_value, torch.Tensor):
            label_tensor = label_value.float()
        elif isinstance(label_value, str):
            stripped = label_value.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                label_tensor = torch.tensor(json.loads(stripped), dtype=torch.float32)
            else:
                label_tensor = torch.tensor(float(label_value), dtype=torch.float32)
        else:
            label_tensor = torch.tensor(label_value, dtype=torch.float32)

        if label_tensor.dim() == 0:
            label_tensor = label_tensor.expand(self.num_frames)
        elif label_tensor.numel() != self.num_frames:
            label_tensor = self._resize_label(label_tensor)
        return label_tensor.float()

    def _resize_label(self, label_tensor: torch.Tensor) -> torch.Tensor:
        flat = label_tensor.flatten().float()
        if flat.numel() == 0:
            return self._build_default_label()
        if flat.numel() >= self.num_frames:
            indices = torch.linspace(0, flat.numel() - 1, self.num_frames).long()
            return flat[indices]
        pad = flat[-1:].repeat(self.num_frames - flat.numel())
        return torch.cat([flat, pad], dim=0)

    def _build_default_label(self) -> torch.Tensor:
        label = torch.zeros(self.num_frames, dtype=torch.float32)
        start = max(0, self.num_frames // 4)
        end = max(start + 1, (3 * self.num_frames) // 4)
        label[start:end] = 1.0
        return label
