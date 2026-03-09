from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import h5py
import numpy as np


SUPPORTED_DATASETS = {"summe", "tvsum"}


@dataclass(slots=True)
class BenchmarkVideoRecord:
    dataset_name: str
    h5_key: str
    video_id: str
    video_name: str
    video_path: str
    fps: float
    n_frames: int
    title: Optional[str] = None
    category: Optional[str] = None
    features: Optional[np.ndarray] = None
    gtscore: Optional[np.ndarray] = None
    gtsummary: Optional[np.ndarray] = None
    change_points: Optional[np.ndarray] = None
    n_frame_per_seg: Optional[np.ndarray] = None
    picks: Optional[np.ndarray] = None
    user_summary: Optional[np.ndarray] = None
    user_scores: Optional[np.ndarray] = None


class BenchmarkDatasetAdapter:
    """Adapter for the real SumMe/TVSum benchmark datasets."""

    def __init__(self, dataset_root: str, dataset_name: str):
        self.dataset_root = os.path.abspath(dataset_root)
        self.dataset_name = self._normalize_dataset_name(dataset_name)
        if self.dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Expected one of {sorted(SUPPORTED_DATASETS)}.")

        self.h5_path = self._resolve_h5_path()
        self.video_dir = self._resolve_video_dir()
        self.info_rows = self._load_tvsum_info_rows() if self.dataset_name == "tvsum" else {}
        self.tvsum_user_scores = self._load_tvsum_user_scores() if self.dataset_name == "tvsum" else {}

        self._index: Dict[str, dict] = {}
        self._aliases: Dict[str, str] = {}
        self._build_index()

    def list_video_ids(self) -> list[str]:
        return sorted(self._index)

    def get_record(self, video_id: str) -> BenchmarkVideoRecord:
        resolved_video_id = self._resolve_public_video_id(video_id)
        entry = self._index[resolved_video_id]

        with h5py.File(self.h5_path, "r") as h5_file:
            group = h5_file[entry["h5_key"]]
            record = BenchmarkVideoRecord(
                dataset_name=self.dataset_name.capitalize() if self.dataset_name == "summe" else "TVSum",
                h5_key=entry["h5_key"],
                video_id=resolved_video_id,
                video_name=entry["video_name"],
                video_path=entry["video_path"],
                fps=entry["fps"],
                n_frames=int(group["n_frames"][()]),
                title=entry.get("title"),
                category=entry.get("category"),
                features=self._read_optional_array(group, "features"),
                gtscore=self._read_optional_array(group, "gtscore"),
                gtsummary=self._read_optional_array(group, "gtsummary"),
                change_points=self._read_optional_array(group, "change_points"),
                n_frame_per_seg=self._read_optional_array(group, "n_frame_per_seg"),
                picks=self._read_optional_array(group, "picks"),
                user_summary=self._read_optional_array(group, "user_summary"),
                user_scores=self.tvsum_user_scores.get(resolved_video_id),
            )

        return record

    def _build_index(self) -> None:
        if self.dataset_name == "summe":
            self._build_summe_index()
            return
        self._build_tvsum_index()

    def _build_summe_index(self) -> None:
        video_lookup = self._build_video_lookup_by_name()
        with h5py.File(self.h5_path, "r") as h5_file:
            for h5_key in sorted(h5_file.keys()):
                group = h5_file[h5_key]
                raw_name = group["video_name"][()]
                video_name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
                video_path = self._lookup_video_path(video_lookup, video_name)
                public_video_id = Path(video_path).stem
                fps = self._probe_video_fps(video_path)
                self._register_entry(
                    public_video_id=public_video_id,
                    entry={
                        "h5_key": h5_key,
                        "video_name": video_name,
                        "video_path": video_path,
                        "fps": fps,
                    },
                    aliases=[h5_key, video_name, public_video_id],
                )

    def _build_tvsum_index(self) -> None:
        frame_lookup = self._build_video_lookup_by_frame_count()
        with h5py.File(self.h5_path, "r") as h5_file:
            for h5_key in sorted(h5_file.keys()):
                group = h5_file[h5_key]
                n_frames = int(group["n_frames"][()])
                video_path = self._lookup_tvsum_video_path(frame_lookup, n_frames)
                public_video_id = Path(video_path).stem
                metadata = self.info_rows.get(public_video_id, {})
                fps = self._probe_video_fps(video_path)
                self._register_entry(
                    public_video_id=public_video_id,
                    entry={
                        "h5_key": h5_key,
                        "video_name": public_video_id,
                        "video_path": video_path,
                        "fps": fps,
                        "title": metadata.get("title"),
                        "category": metadata.get("category"),
                    },
                    aliases=[h5_key, public_video_id, metadata.get("title") or ""],
                )

    def _register_entry(self, public_video_id: str, entry: dict, aliases: list[str]) -> None:
        self._index[public_video_id] = entry
        for alias in aliases:
            normalized = self._normalize_key(alias)
            if normalized:
                self._aliases[normalized] = public_video_id

    def _resolve_public_video_id(self, video_id: str) -> str:
        normalized = self._normalize_key(video_id)
        if normalized in self._aliases:
            return self._aliases[normalized]
        available = ", ".join(self.list_video_ids()[:10])
        raise KeyError(f"Unknown video_id={video_id!r}. First available ids: {available}")

    def _resolve_h5_path(self) -> str:
        file_name = "eccv16_dataset_summe_google_pool5.h5" if self.dataset_name == "summe" else "eccv16_dataset_tvsum_google_pool5.h5"
        resolved = os.path.join(self.dataset_root, file_name)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"H5 file not found: {resolved}")
        return resolved

    def _resolve_video_dir(self) -> str:
        if self.dataset_name == "summe":
            resolved = os.path.join(self.dataset_root, "SumMe", "videos")
        else:
            resolved = os.path.join(self.dataset_root, "TVSum", "ydata-tvsum50-v1_1", "video")
        if not os.path.isdir(resolved):
            raise FileNotFoundError(f"Video directory not found: {resolved}")
        return resolved

    def _build_video_lookup_by_name(self) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for file_name in sorted(os.listdir(self.video_dir), key=self._video_sort_key):
            full_path = os.path.join(self.video_dir, file_name)
            if not os.path.isfile(full_path):
                continue
            stem = Path(file_name).stem
            normalized = self._normalize_key(stem)
            if normalized not in lookup:
                lookup[normalized] = full_path
        return lookup

    def _build_video_lookup_by_frame_count(self) -> Dict[int, list[str]]:
        lookup: Dict[int, list[str]] = {}
        for file_name in sorted(os.listdir(self.video_dir)):
            full_path = os.path.join(self.video_dir, file_name)
            if not os.path.isfile(full_path):
                continue
            frame_count = self._probe_video_frame_count(full_path)
            lookup.setdefault(frame_count, []).append(full_path)
        return lookup

    def _lookup_video_path(self, lookup: Dict[str, str], video_name: str) -> str:
        normalized = self._normalize_key(video_name)
        if normalized in lookup:
            return lookup[normalized]
        raise FileNotFoundError(f"Failed to resolve video path for {video_name!r} in {self.video_dir}")

    def _lookup_tvsum_video_path(self, frame_lookup: Dict[int, list[str]], n_frames: int) -> str:
        exact_matches = frame_lookup.get(n_frames, [])
        if len(exact_matches) == 1:
            return exact_matches[0]

        nearby_matches: list[str] = []
        for delta in (1, -1, 2, -2):
            nearby_matches.extend(frame_lookup.get(n_frames + delta, []))
        unique_matches = sorted(dict.fromkeys(nearby_matches))
        if len(unique_matches) == 1:
            return unique_matches[0]

        raise FileNotFoundError(
            f"Failed to resolve TVSum video with n_frames={n_frames}. Exact={exact_matches}, nearby={unique_matches}."
        )

    def _load_tvsum_info_rows(self) -> Dict[str, dict]:
        info_path = os.path.join(self.dataset_root, "TVSum", "ydata-tvsum50-v1_1", "data", "ydata-tvsum50-info.tsv")
        if not os.path.exists(info_path):
            return {}

        with open(info_path, "r", encoding="utf-8") as file_obj:
            reader = csv.DictReader(file_obj, delimiter="\t")
            return {
                str(row["video_id"]): {
                    "title": row.get("title") or None,
                    "category": row.get("category") or None,
                }
                for row in reader
            }

    def _load_tvsum_user_scores(self) -> Dict[str, np.ndarray]:
        anno_path = os.path.join(self.dataset_root, "TVSum", "ydata-tvsum50-v1_1", "data", "ydata-tvsum50-anno.tsv")
        if not os.path.exists(anno_path):
            return {}

        rows_by_video: Dict[str, list[np.ndarray]] = {}
        with open(anno_path, "r", encoding="utf-8") as file_obj:
            reader = csv.reader(file_obj, delimiter="\t")
            for row in reader:
                if len(row) < 3:
                    continue
                video_id = str(row[0])
                scores = np.asarray([float(item) for item in row[2].split(",")], dtype=np.float32)
                rows_by_video.setdefault(video_id, []).append(scores)

        return {
            video_id: np.stack(user_rows, axis=1)
            for video_id, user_rows in rows_by_video.items()
            if user_rows
        }

    def _read_optional_array(self, group: h5py.Group, key: str) -> Optional[np.ndarray]:
        if key not in group:
            return None
        return np.asarray(group[key][()])

    def _probe_video_frame_count(self, video_path: str) -> int:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        try:
            return int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            capture.release()

    def _probe_video_fps(self, video_path: str) -> float:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        finally:
            capture.release()
        if fps <= 0:
            raise ValueError(f"Invalid fps for video file: {video_path}")
        return fps

    def _normalize_dataset_name(self, dataset_name: str) -> str:
        return str(dataset_name).strip().lower()

    def _normalize_key(self, value: str) -> str:
        return "".join(ch.lower() for ch in str(value) if ch.isalnum())

    def _video_sort_key(self, file_name: str) -> tuple[int, str]:
        suffix = Path(file_name).suffix.lower()
        priority = 0 if suffix == ".mp4" else 1
        return (priority, file_name.lower())
