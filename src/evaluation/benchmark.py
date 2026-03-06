from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.metrics import (
    coverage_score,
    diversity_score,
    kendall_correlation,
    latency_statistics,
    normalize_pred_scores,
    precision_recall_fscore,
    spearman_correlation,
)


class EvaluationBenchmark:
    """Evaluate summary outputs and save metrics with visualization artifacts."""

    def evaluate(
        self,
        predicted_scores: Sequence[str | int | float],
        gt_scores: Sequence[int | float],
        selected_segments: Sequence[dict],
        latencies_sec: Optional[Iterable[float]] = None,
    ) -> Dict[str, Any]:
        metrics = precision_recall_fscore(predicted_scores=predicted_scores, gt_scores=gt_scores)
        metrics.update(
            {
                "spearman": spearman_correlation(predicted_scores=predicted_scores, gt_scores=gt_scores),
                "kendall": kendall_correlation(predicted_scores=predicted_scores, gt_scores=gt_scores),
                "coverage": coverage_score(predicted_scores=predicted_scores, gt_scores=gt_scores),
                "diversity": diversity_score(selected_segments),
            }
        )
        metrics["latency"] = latency_statistics(latencies_sec or [])
        return metrics

    def save_report(
        self,
        metrics: Dict[str, Any],
        predicted_scores: Sequence[str | int | float],
        gt_scores: Sequence[int | float],
        output_dir: str,
        video_id: str = "evaluation",
    ) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)

        metrics_path = os.path.join(output_dir, f"{video_id}_results.json")
        plot_path = os.path.join(output_dir, f"{video_id}_score_curve.png")

        with open(metrics_path, "w", encoding="utf-8") as file_obj:
            json.dump(metrics, file_obj, ensure_ascii=False, indent=2)

        self.plot_score_curve(
            predicted_scores=predicted_scores,
            gt_scores=gt_scores,
            output_path=plot_path,
            title=f"Score Curve - {video_id}",
        )

        return {"metrics_path": metrics_path, "plot_path": plot_path}

    def plot_score_curve(
        self,
        predicted_scores: Sequence[str | int | float],
        gt_scores: Sequence[int | float],
        output_path: str,
        title: str = "Score Curve",
    ) -> None:
        pred_values = normalize_pred_scores(predicted_scores)
        gt_values = [float(item) for item in gt_scores]
        frame_indices = list(range(len(pred_values)))

        plt.figure(figsize=(10, 4))
        plt.plot(frame_indices, pred_values, label="predicted", linewidth=2)
        plt.plot(frame_indices, gt_values, label="ground_truth", linewidth=2)
        plt.xlabel("Frame Index")
        plt.ylabel("Importance Score")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()