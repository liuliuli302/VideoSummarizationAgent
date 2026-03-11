from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.data import DatasetVideoRecord, EvalOverview, EvalVariantResult
from src.io import JsonSaver


class EvaluationReporter:
    def __init__(self, output_root: str = "outputs/evaluation", exam_name: str | None = None) -> None:
        self.output_root = output_root
        self.exam_name = self._normalize_exam_name(exam_name)
        self.exam_dir = os.path.join(self.output_root, self.exam_name)
        self.json_saver = JsonSaver()

    def get_video_dir(self, dataset_name: str, video_id: str) -> str:
        video_name = Path(video_id).stem
        return os.path.join(self.exam_dir, dataset_name, video_name)

    def normalize_scores(self, frame_scores: list[float]) -> list[float]:
        scores = np.asarray(frame_scores, dtype=np.float32)
        if scores.size == 0:
            return []
        min_value = float(scores.min())
        max_value = float(scores.max())
        if max_value - min_value <= 1e-8:
            return [0.0 for _ in range(scores.size)]
        normalized = (scores - min_value) / (max_value - min_value)
        return normalized.astype(float).tolist()

    def smooth_scores(self, frame_scores: list[float], window_size: int = 5) -> list[float]:
        scores = np.asarray(frame_scores, dtype=np.float32)
        if scores.size == 0:
            return []
        effective_window = max(1, int(window_size))
        effective_window = min(effective_window, scores.size)
        if effective_window == 1:
            return scores.astype(float).tolist()
        if effective_window % 2 == 0:
            effective_window += 1 if effective_window < scores.size else -1
        effective_window = max(1, effective_window)
        pad = effective_window // 2
        padded = np.pad(scores, (pad, pad), mode="edge")
        kernel = np.ones(effective_window, dtype=np.float32) / float(effective_window)
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed.astype(float).tolist()

    def build_gt_curve(self, record: DatasetVideoRecord) -> list[float]:
        if record.user_scores:
            scores = np.asarray(record.user_scores, dtype=np.float32)
            reduced = scores if scores.ndim == 1 else scores.mean(axis=0)
        else:
            summaries = np.asarray(record.user_summary, dtype=np.float32)
            reduced = summaries.mean(axis=0) if summaries.ndim > 1 else summaries
        reduced = reduced.astype(np.float32).ravel()
        if reduced.size == 0:
            return []
        min_value = float(reduced.min())
        max_value = float(reduced.max())
        if max_value - min_value <= 1e-8:
            return reduced.astype(float).tolist()
        normalized = (reduced - min_value) / (max_value - min_value)
        return normalized.astype(float).tolist()

    def save_video_report(
        self,
        record: DatasetVideoRecord,
        variants: list[EvalVariantResult],
        original_frame_scores: list[float],
        smooth_window_size: int,
    ) -> dict[str, str]:
        video_dir = self.get_video_dir(record.dataset_name, record.video_id)
        gt_curve = self.build_gt_curve(record)
        plot_path = os.path.join(video_dir, "frame_scores_vs_gt.png")

        variants_payload = [variant.to_dict() for variant in variants]
        self.json_saver.save(
            os.path.join(video_dir, "frame_scores_variants.json"),
            {
                "exam_name": self.exam_name,
                "dataset_name": record.dataset_name,
                "video_id": record.video_id,
                "smooth_window_size": smooth_window_size,
                "original_frame_scores": original_frame_scores,
                "gt_curve": gt_curve,
                "variants": variants_payload,
            },
        )

        for variant in variants:
            self.json_saver.save(
                os.path.join(video_dir, f"eval_{variant.variant_name}.json"),
                variant.eval_result.to_dict(),
            )

        self._save_plot(
            plot_path=plot_path,
            gt_curve=gt_curve,
            raw_scores=self._get_variant_scores(variants, "normalized_raw"),
            smoothed_scores=self._get_variant_scores(variants, "normalized_smoothed"),
            title=f"{record.dataset_name} / {Path(record.video_id).stem}",
        )

        overview = self._update_exam_overview()
        return {
            "exam_dir": self.exam_dir,
            "video_dir": video_dir,
            "plot_path": plot_path,
            "overview_path": os.path.join(self.exam_dir, "overview.json"),
            "overview_markdown_path": os.path.join(self.exam_dir, "overview.md"),
            "overview_records_path": os.path.join(self.exam_dir, "overview_records.json"),
            "total_videos": str(overview.total_videos),
        }

    def _save_plot(
        self,
        plot_path: str,
        gt_curve: list[float],
        raw_scores: list[float],
        smoothed_scores: list[float],
        title: str,
    ) -> None:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.figure(figsize=(12, 4.8))
        if gt_curve:
            plt.plot(gt_curve, label="gt", linewidth=2.0, color="#1f1f1f", alpha=0.85)
        if raw_scores:
            plt.plot(raw_scores, label="pred_normalized", linewidth=1.4, color="#d95f02", alpha=0.8)
        if smoothed_scores:
            plt.plot(smoothed_scores, label="pred_smoothed", linewidth=1.8, color="#1b9e77", alpha=0.9)
        plt.title(title)
        plt.xlabel("Frame Index")
        plt.ylabel("Normalized Importance")
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=180)
        plt.close()

    def _get_variant_scores(self, variants: list[EvalVariantResult], variant_name: str) -> list[float]:
        for variant in variants:
            if variant.variant_name == variant_name:
                return variant.frame_scores
        return []

    def _update_exam_overview(self) -> EvalOverview:
        records_path = os.path.join(self.exam_dir, "overview_records.json")
        records = self._collect_exam_records()
        overview = EvalOverview.build(exam_name=self.exam_name, records=records)
        self.json_saver.save(records_path, records)
        self.json_saver.save(os.path.join(self.exam_dir, "overview.json"), overview.to_dict())
        self._save_overview_markdown(overview)
        return overview

    def _collect_exam_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if not os.path.isdir(self.exam_dir):
            return records
        for dataset_name in sorted(os.listdir(self.exam_dir)):
            dataset_dir = os.path.join(self.exam_dir, dataset_name)
            if not os.path.isdir(dataset_dir):
                continue
            for video_name in sorted(os.listdir(dataset_dir)):
                video_dir = os.path.join(dataset_dir, video_name)
                variants_path = os.path.join(video_dir, "frame_scores_variants.json")
                if not os.path.isfile(variants_path):
                    continue
                with open(variants_path, "r", encoding="utf-8") as file_obj:
                    payload = __import__("json").load(file_obj)
                for variant in payload.get("variants", []):
                    eval_payload = variant.get("eval_result", {})
                    records.append(
                        {
                            "dataset_name": payload.get("dataset_name", dataset_name),
                            "video_id": payload.get("video_id", video_name),
                            "variant_name": variant.get("variant_name", "unknown"),
                            "f1": float(eval_payload.get("f1", 0.0)),
                            "precision": float(eval_payload.get("precision", 0.0)),
                            "recall": float(eval_payload.get("recall", 0.0)),
                            "rho": float(eval_payload.get("rho", 0.0)),
                            "tau": float(eval_payload.get("tau", 0.0)),
                        }
                    )
        return records

    def _save_overview_markdown(self, overview: EvalOverview) -> None:
        lines = [
            f"# {overview.exam_name}",
            "",
            f"Total videos: {overview.total_videos}",
            "",
            "## Overall",
            "",
            "| Variant | Count | F1 | Precision | Recall | Rho | Tau |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for variant_name, metrics in sorted(overview.per_variant_metrics.items()):
            lines.append(
                "| {variant} | {count:.0f} | {f1:.4f} | {precision:.4f} | {recall:.4f} | {rho:.4f} | {tau:.4f} |".format(
                    variant=variant_name,
                    count=metrics.get("count", 0.0),
                    f1=metrics.get("f1_mean", 0.0),
                    precision=metrics.get("precision_mean", 0.0),
                    recall=metrics.get("recall_mean", 0.0),
                    rho=metrics.get("rho_mean", 0.0),
                    tau=metrics.get("tau_mean", 0.0),
                )
            )
        lines.extend(["", "## By Dataset", ""])
        for dataset_name, variants in sorted(overview.per_dataset_metrics.items()):
            lines.extend([
                f"### {dataset_name}",
                "",
                "| Variant | Count | F1 | Precision | Recall | Rho | Tau |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ])
            for variant_name, metrics in sorted(variants.items()):
                lines.append(
                    "| {variant} | {count:.0f} | {f1:.4f} | {precision:.4f} | {recall:.4f} | {rho:.4f} | {tau:.4f} |".format(
                        variant=variant_name,
                        count=metrics.get("count", 0.0),
                        f1=metrics.get("f1_mean", 0.0),
                        precision=metrics.get("precision_mean", 0.0),
                        recall=metrics.get("recall_mean", 0.0),
                        rho=metrics.get("rho_mean", 0.0),
                        tau=metrics.get("tau_mean", 0.0),
                    )
                )
            lines.append("")
        markdown_path = os.path.join(self.exam_dir, "overview.md")
        os.makedirs(self.exam_dir, exist_ok=True)
        with open(markdown_path, "w", encoding="utf-8") as file_obj:
            file_obj.write("\n".join(lines).rstrip() + "\n")

    def _normalize_exam_name(self, exam_name: str | None) -> str:
        if exam_name:
            return exam_name if exam_name.startswith("exam_") else f"exam_{exam_name}"
        return datetime.now().strftime("exam_%Y%m%d_%H%M%S")