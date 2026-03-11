from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from src.io import JsonSaver


class SplitEvaluationAggregator:
    def __init__(self, split_root: str, split_count: int = 5) -> None:
        self.split_root = split_root
        self.split_count = split_count
        self.json_saver = JsonSaver()

    def aggregate_exam(self, exam_dir: str, datasets: list[str]) -> dict[str, Any]:
        summary = {
            "split_root": self.split_root,
            "split_count": self.split_count,
            "datasets": {},
        }
        for dataset_name in datasets:
            summary["datasets"][dataset_name] = self.aggregate_exam_dataset(exam_dir, dataset_name)

        output_path = os.path.join(exam_dir, "split_overview.json")
        self.json_saver.save(output_path, summary)
        self._save_markdown(os.path.join(exam_dir, "split_overview.md"), summary)
        return summary

    def aggregate_exam_dataset(self, exam_dir: str, dataset_name: str) -> dict[str, Any]:
        records = self._load_exam_records(exam_dir, dataset_name)
        variant_names = sorted({record["variant_name"] for record in records})
        split_mapping = self._load_mapping(dataset_name)
        splits = self._load_splits(dataset_name)
        normalized_record_map = {
            (self._normalize_video_name(record["video_id"]), record["variant_name"]): record for record in records
        }

        per_variant_split_results: dict[str, list[dict[str, Any]]] = {name: [] for name in variant_names}
        for split_id, split_info in enumerate(splits, start=1):
            test_keys = list(split_info.get("test_keys", []))
            resolved_video_ids = [split_mapping[key] for key in test_keys if key in split_mapping]
            normalized_video_ids = [self._normalize_video_name(video_id) for video_id in resolved_video_ids]

            for variant_name in variant_names:
                matched_records = [
                    normalized_record_map[(video_id, variant_name)]
                    for video_id in normalized_video_ids
                    if (video_id, variant_name) in normalized_record_map
                ]
                per_variant_split_results[variant_name].append(
                    {
                        "split_id": split_id,
                        "test_keys": test_keys,
                        "video_ids": resolved_video_ids,
                        "num_videos": len(matched_records),
                        **self._aggregate_records(matched_records),
                    }
                )

        return {
            "dataset_name": dataset_name,
            "split_file": os.path.join(self.split_root, f"{dataset_name}_splits_{self.split_count}.json"),
            "mapping_file": os.path.join(self.split_root, f"{dataset_name}_mapping.json"),
            "variants": {
                variant_name: {
                    **self._aggregate_split_results(split_results),
                    "split_results": split_results,
                }
                for variant_name, split_results in per_variant_split_results.items()
            },
        }

    def _load_exam_records(self, exam_dir: str, dataset_name: str) -> list[dict[str, Any]]:
        records_path = os.path.join(exam_dir, "overview_records.json")
        if not os.path.isfile(records_path):
            return []
        with open(records_path, "r", encoding="utf-8") as file_obj:
            records = json.load(file_obj)
        return [record for record in records if str(record.get("dataset_name", "")).lower() == dataset_name.lower()]

    def _load_mapping(self, dataset_name: str) -> dict[str, str]:
        mapping_path = os.path.join(self.split_root, f"{dataset_name}_mapping.json")
        with open(mapping_path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    def _load_splits(self, dataset_name: str) -> list[dict[str, Any]]:
        split_path = os.path.join(self.split_root, f"{dataset_name}_splits_{self.split_count}.json")
        with open(split_path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    def _aggregate_records(self, records: list[dict[str, Any]]) -> dict[str, float]:
        if not records:
            return {
                "mean_f1": 0.0,
                "mean_precision": 0.0,
                "mean_recall": 0.0,
                "mean_rho": 0.0,
                "mean_tau": 0.0,
            }
        return {
            "mean_f1": float(mean(float(record.get("f1", 0.0)) for record in records)),
            "mean_precision": float(mean(float(record.get("precision", 0.0)) for record in records)),
            "mean_recall": float(mean(float(record.get("recall", 0.0)) for record in records)),
            "mean_rho": float(mean(float(record.get("rho", 0.0)) for record in records)),
            "mean_tau": float(mean(float(record.get("tau", 0.0)) for record in records)),
        }

    def _aggregate_split_results(self, split_results: list[dict[str, Any]]) -> dict[str, Any]:
        if not split_results:
            return {
                "num_splits": 0,
                "num_videos_avg_per_split": 0.0,
                "mean_f1": 0.0,
                "mean_precision": 0.0,
                "mean_recall": 0.0,
                "mean_rho": 0.0,
                "mean_tau": 0.0,
                "best_split": None,
            }
        best_split = sorted(
            split_results,
            key=lambda item: (item["mean_f1"], item["mean_rho"], item["mean_tau"]),
            reverse=True,
        )[0]
        return {
            "num_splits": len(split_results),
            "num_videos_avg_per_split": float(mean(float(item["num_videos"]) for item in split_results)),
            "mean_f1": float(mean(float(item["mean_f1"]) for item in split_results)),
            "mean_precision": float(mean(float(item["mean_precision"]) for item in split_results)),
            "mean_recall": float(mean(float(item["mean_recall"]) for item in split_results)),
            "mean_rho": float(mean(float(item["mean_rho"]) for item in split_results)),
            "mean_tau": float(mean(float(item["mean_tau"]) for item in split_results)),
            "best_split": best_split,
        }

    def _save_markdown(self, markdown_path: str, summary: dict[str, Any]) -> None:
        lines = [
            f"# Split Overview ({summary['split_count']} splits)",
            "",
            f"Split root: {summary['split_root']}",
            "",
        ]
        for dataset_name, dataset_summary in sorted(summary["datasets"].items()):
            lines.extend([
                f"## {dataset_name}",
                "",
                "| Variant | Mean F1 | Mean Precision | Mean Recall | Mean Rho | Mean Tau | Avg Videos/Split | Splits |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ])
            for variant_name, variant_summary in sorted(dataset_summary.get("variants", {}).items()):
                lines.append(
                    "| {variant} | {f1:.4f} | {precision:.4f} | {recall:.4f} | {rho:.4f} | {tau:.4f} | {videos:.2f} | {splits} |".format(
                        variant=variant_name,
                        f1=variant_summary.get("mean_f1", 0.0),
                        precision=variant_summary.get("mean_precision", 0.0),
                        recall=variant_summary.get("mean_recall", 0.0),
                        rho=variant_summary.get("mean_rho", 0.0),
                        tau=variant_summary.get("mean_tau", 0.0),
                        videos=variant_summary.get("num_videos_avg_per_split", 0.0),
                        splits=variant_summary.get("num_splits", 0),
                    )
                )
            lines.append("")
        with open(markdown_path, "w", encoding="utf-8") as file_obj:
            file_obj.write("\n".join(lines).rstrip() + "\n")

    def _normalize_video_name(self, value: str) -> str:
        return Path(str(value)).stem.strip().lower()