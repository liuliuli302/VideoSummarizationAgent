from __future__ import annotations

import numpy as np

from src.data.schemas import DatasetVideoRecord, EvalResult
from src.evaluation.vsum_utils import (
    build_frame_summary_from_segments,
    evaluate_f1_frame_summary,
    evaluate_rank_correlation,
)


class VsumEvaluator:
    def evaluate(self, frame_scores: list[float], record: DatasetVideoRecord) -> EvalResult:
        predicted_scores = np.asarray(frame_scores, dtype=np.float32)
        picks = np.asarray(record.picks, dtype=np.int32)
        if predicted_scores.size != picks.size:
            raise ValueError(
                f"Predicted frame_scores length {predicted_scores.size} does not match picks length {picks.size}."
            )

        summary, _, _ = build_frame_summary_from_segments(
            predicted_scores=predicted_scores,
            change_points=np.asarray(record.change_points, dtype=np.int32),
            total_frames=int(record.n_frames),
            frames_per_segment=np.asarray(record.n_frame_per_seg, dtype=np.int32).tolist(),
            sampled_positions=picks,
            summary_ratio=0.15,
            selection_method="knapsack",
        )

        reduction = "max" if record.dataset_name == "summe" else "avg"
        f1, precision, recall = evaluate_f1_frame_summary(
            machine_summary=summary,
            human_summaries=np.asarray(record.user_summary, dtype=np.float32),
            reduction=reduction,
        )

        human_scores = np.asarray(record.user_scores if record.user_scores else record.user_summary, dtype=np.float32)
        rho, tau = evaluate_rank_correlation(
            predicted_scores=predicted_scores,
            human_scores=human_scores,
            reduction="avg",
        )

        return EvalResult(
            dataset_name=record.dataset_name,
            video_id=record.video_id,
            f1=float(f1),
            precision=float(precision),
            recall=float(recall),
            rho=float(rho),
            tau=float(tau),
            selected_summary=np.asarray(summary, dtype=np.float32).astype(float).tolist(),
        )