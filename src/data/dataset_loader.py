from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np

from src.data.schemas import DatasetVideoRecord


class DatasetLoader:
    def __init__(self, dataset_root: str) -> None:
        self.dataset_root = os.path.abspath(dataset_root)

    def list_video_ids(self, dataset_name: str) -> list[str]:
        normalized_dataset = dataset_name.lower()
        if normalized_dataset not in {"summe", "tvsum"}:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        mapping = self._load_mapping(normalized_dataset)
        return [mapping[key] for key in self._sorted_mapping_keys(mapping.keys())]

    def load_record(self, dataset_name: str, video_id: str) -> DatasetVideoRecord:
        normalized_dataset = dataset_name.lower()
        if normalized_dataset not in {"summe", "tvsum"}:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        h5_path = self._resolve_h5_path(normalized_dataset)
        mapping = self._load_mapping(normalized_dataset)
        video_key = self._find_video_key(mapping, video_id)
        mapped_name = mapping[video_key]
        video_path = self._resolve_video_path(normalized_dataset, mapped_name)

        with h5py.File(h5_path, "r") as h5_file:
            group = h5_file[video_key]
            return DatasetVideoRecord(
                dataset_name=normalized_dataset,
                video_id=mapped_name,
                video_path=video_path,
                n_frames=int(group["n_frames"][()]),
                picks=self._to_int_list(group["picks"][()]),
                change_points=self._to_nested_int_list(group["change_points"][()]),
                n_frame_per_seg=self._to_int_list(group["n_frame_per_seg"][()]),
                user_summary=self._to_nested_float_list(group["user_summary"][()]),
                user_scores=self._extract_user_scores(group),
            )

    def _resolve_h5_path(self, dataset_name: str) -> str:
        file_name = (
            "eccv16_dataset_summe_google_pool5.h5"
            if dataset_name == "summe"
            else "eccv16_dataset_tvsum_google_pool5.h5"
        )
        return os.path.join(self.dataset_root, file_name)

    def _load_mapping(self, dataset_name: str) -> dict[str, str]:
        mapping_path = os.path.join(self.dataset_root, f"{dataset_name}_mapping.json")
        with open(mapping_path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    def _sorted_mapping_keys(self, keys) -> list[str]:
        def _sort_key(value: str) -> tuple[int, str]:
            stem = Path(value).stem.lower()
            if stem.startswith("video_"):
                suffix = stem.split("_", 1)[1]
                if suffix.isdigit():
                    return int(suffix), stem
            return 10**9, stem

        return sorted(keys, key=_sort_key)

    def _find_video_key(self, mapping: dict[str, str], video_id: str) -> str:
        normalized_target = Path(video_id).stem.lower()
        for key, value in mapping.items():
            if key.lower() == normalized_target or Path(value).stem.lower() == normalized_target:
                return key
        raise KeyError(f"Failed to find video_id={video_id} in dataset mapping.")

    def _resolve_video_path(self, dataset_name: str, mapped_name: str) -> str:
        if dataset_name == "summe":
            video_dir = os.path.join(self.dataset_root, "SumMe", "videos")
        else:
            video_dir = os.path.join(self.dataset_root, "TVSum", "ydata-tvsum50-v1_1", "video")

        target_stem = Path(mapped_name).stem.lower()
        for file_name in os.listdir(video_dir):
            full_path = os.path.join(video_dir, file_name)
            if not os.path.isfile(full_path):
                continue
            if Path(file_name).stem.lower() == target_stem:
                return full_path

        raise FileNotFoundError(f"Failed to resolve video path for {mapped_name} under {video_dir}")

    def _extract_user_scores(self, group: h5py.Group) -> list[list[float]] | list[float]:
        if "user_scores" in group:
            return self._to_nested_float_list(group["user_scores"][()])
        if "gtscore" in group:
            array = np.asarray(group["gtscore"][()], dtype=np.float32)
            return array.astype(float).tolist()
        return []

    def _to_int_list(self, values: np.ndarray) -> list[int]:
        return np.asarray(values, dtype=np.int32).astype(int).tolist()

    def _to_nested_int_list(self, values: np.ndarray) -> list[list[int]]:
        return np.asarray(values, dtype=np.int32).astype(int).tolist()

    def _to_nested_float_list(self, values: np.ndarray) -> list[list[float]]:
        array = np.asarray(values, dtype=np.float32)
        return array.astype(float).tolist()