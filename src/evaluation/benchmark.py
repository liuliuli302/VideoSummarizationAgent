from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional, Sequence

from src.evaluation.official_protocol import evaluate_benchmark_video
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

    def evaluate_dataset_record(
        self,
        record: Any,
        predicted_scores: Sequence[str | int | float],
        scene_ranges: Sequence[dict | tuple[int, int]],
        budget_ratio: float,
    ) -> Dict[str, Any]:
        report = evaluate_benchmark_video(
            dataset_name=str(record.dataset_name),
            predicted_scores=predicted_scores,
            scene_ranges=scene_ranges,
            n_frames=int(record.n_frames),
            budget_ratio=float(budget_ratio),
            user_summary=getattr(record, "user_summary", None),
            user_scores=getattr(record, "user_scores", None),
        )

        gtscore = getattr(record, "gtscore", None)
        picks = getattr(record, "picks", None)
        if gtscore is not None and picks is not None:
            sampled_pred = self._sample_scores_at_picks(predicted_scores, picks=picks, n_frames=int(record.n_frames))
            sampled_gt = [float(item) for item in gtscore]
            report["sampled_score_alignment"] = {
                "spearman": spearman_correlation(sampled_pred, sampled_gt),
                "kendall": kendall_correlation(sampled_pred, sampled_gt),
                "coverage": coverage_score(sampled_pred, sampled_gt),
            }
            report["sampled_predicted_scores"] = sampled_pred
            report["sampled_gt_scores"] = sampled_gt
        return report

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

        try:
            self.plot_score_curve(
                predicted_scores=predicted_scores,
                gt_scores=gt_scores,
                output_path=plot_path,
                title=f"Score Curve - {video_id}",
            )
        except Exception as exc:  # pragma: no cover - defensive fallback for mixed environments
            metrics["plot_warning"] = f"plot generation skipped: {exc}"
            with open(metrics_path, "w", encoding="utf-8") as file_obj:
                json.dump(metrics, file_obj, ensure_ascii=False, indent=2)
            with open(plot_path, "wb") as file_obj:
                file_obj.write(b"")

        return {"metrics_path": metrics_path, "plot_path": plot_path}

    def _sample_scores_at_picks(
        self,
        predicted_scores: Sequence[str | int | float],
        picks: Sequence[int],
        n_frames: int,
    ) -> list[float]:
        numeric_scores = normalize_pred_scores(predicted_scores)
        if not numeric_scores:
            return []

        bounded_scores = numeric_scores[:n_frames]
        if len(bounded_scores) < n_frames:
            bounded_scores.extend([bounded_scores[-1]] * (n_frames - len(bounded_scores)))

        sampled = []
        for raw_pick in picks:
            pick = max(0, min(int(raw_pick), max(0, n_frames - 1)))
            sampled.append(float(bounded_scores[pick]))
        return sampled

    def plot_score_curve(
        self,
        predicted_scores: Sequence[str | int | float],
        gt_scores: Sequence[int | float],
        output_path: str,
        title: str = "Score Curve",
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

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