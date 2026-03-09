from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from src.evaluation.metrics import normalize_pred_scores


def build_summary_from_segments(
    frame_scores: Sequence[str | int | float],
    segments: Sequence[tuple[int, int]],
    n_frames: int,
    budget_ratio: float,
) -> np.ndarray:
    """Reproduce the official segment-budgeted summary generation logic."""
    if n_frames < 0:
        raise ValueError(f"n_frames must be non-negative, got {n_frames}.")
    if not 0 <= budget_ratio <= 1:
        raise ValueError(f"budget_ratio must be in [0, 1], got {budget_ratio}.")
    if n_frames == 0:
        return np.zeros((0,), dtype=np.int32)

    numeric_scores = np.asarray(normalize_pred_scores(frame_scores), dtype=np.float32)
    numeric_scores = _resize_vector(numeric_scores, n_frames)
    bounded_segments = _sanitize_segments(segments, n_frames)
    budget = int(np.fix(budget_ratio * n_frames))

    if not bounded_segments or budget <= 0:
        return _fallback_summary(numeric_scores, budget_ratio)

    seg_scores = np.asarray([
        float(np.mean(numeric_scores[start:end])) if end > start else 0.0
        for start, end in bounded_segments
    ], dtype=np.float32)
    seg_weights = np.asarray([max(0, end - start) for start, end in bounded_segments], dtype=np.int32)

    selected_segments = _solve_knapsack(seg_weights, seg_scores, budget)
    machine_summary = np.zeros((n_frames,), dtype=np.int32)
    for is_selected, (start, end) in zip(selected_segments, bounded_segments):
        if is_selected:
            machine_summary[start:end] = 1

    if np.unique(machine_summary).size == 1 and machine_summary[0] == 0:
        return _fallback_summary(numeric_scores, budget_ratio)
    return machine_summary


def evaluate_summe(machine_summary: Sequence[int | float], user_summary: np.ndarray) -> dict:
    machine = _resize_binary(machine_summary, int(user_summary.shape[-1]))
    user_matrix = _normalize_user_summary(user_summary)

    precisions = []
    recalls = []
    f1_scores = []
    for gt_summary in user_matrix:
        precision, recall, f1 = _binary_prf(machine, gt_summary)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "precision": round(float(np.mean(precisions)) if precisions else 0.0, 4),
        "recall": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        "fscore": round(float(np.mean(f1_scores)) if f1_scores else 0.0, 4),
        "max_fscore": round(float(np.max(f1_scores)) if f1_scores else 0.0, 4),
        "summary_length": round(float(np.mean(machine)) if machine.size else 0.0, 4),
        "num_users": int(user_matrix.shape[0]),
    }


def evaluate_tvsum(
    machine_summary: Sequence[int | float],
    user_scores: np.ndarray,
    segments: Sequence[tuple[int, int]],
    n_frames: int,
    budget_ratio: float,
) -> dict:
    machine = _resize_binary(machine_summary, n_frames)
    user_score_matrix = _normalize_user_scores(user_scores, n_frames)

    precisions = []
    recalls = []
    f1_scores = []
    for user_idx in range(user_score_matrix.shape[1]):
        gt_summary = build_summary_from_segments(
            frame_scores=user_score_matrix[:, user_idx],
            segments=segments,
            n_frames=n_frames,
            budget_ratio=budget_ratio,
        )
        precision, recall, f1 = _binary_prf(machine, gt_summary)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "precision": round(float(np.mean(precisions)) if precisions else 0.0, 4),
        "recall": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        "fscore": round(float(np.mean(f1_scores)) if f1_scores else 0.0, 4),
        "summary_length": round(float(np.mean(machine)) if machine.size else 0.0, 4),
        "num_users": int(user_score_matrix.shape[1]),
    }


def evaluate_benchmark_video(
    dataset_name: str,
    predicted_scores: Sequence[str | int | float],
    scene_ranges: Sequence[dict | tuple[int, int]],
    n_frames: int,
    budget_ratio: float,
    user_summary: np.ndarray | None = None,
    user_scores: np.ndarray | None = None,
) -> dict:
    segments = _scene_ranges_to_segments(scene_ranges, n_frames)
    machine_summary = build_summary_from_segments(
        frame_scores=predicted_scores,
        segments=segments,
        n_frames=n_frames,
        budget_ratio=budget_ratio,
    )
    dataset_key = str(dataset_name).strip().lower()

    if dataset_key == "summe":
        if user_summary is None:
            raise ValueError("SumMe evaluation requires user_summary.")
        metrics = evaluate_summe(machine_summary=machine_summary, user_summary=user_summary)
    elif dataset_key == "tvsum":
        if user_scores is None:
            raise ValueError("TVSum evaluation requires raw user_scores from the official annotations.")
        metrics = evaluate_tvsum(
            machine_summary=machine_summary,
            user_scores=user_scores,
            segments=segments,
            n_frames=n_frames,
            budget_ratio=budget_ratio,
        )
    else:
        raise ValueError(f"Unsupported dataset for evaluation: {dataset_name}")

    return {
        "metrics": metrics,
        "machine_summary": machine_summary.astype(int).tolist(),
        "n_frames": int(n_frames),
        "budget_ratio": float(budget_ratio),
        "segments": [{"start_frame": start, "end_frame": end} for start, end in segments],
    }


def _solve_knapsack(weights: np.ndarray, values: np.ndarray, capacity: int) -> list[int]:
    n_items = int(len(weights))
    if n_items == 0 or capacity <= 0:
        return [0 for _ in range(n_items)]

    dp = np.zeros((n_items + 1, capacity + 1), dtype=np.float32)
    keep = np.zeros((n_items + 1, capacity + 1), dtype=np.int8)

    for item_idx in range(1, n_items + 1):
        weight = int(weights[item_idx - 1])
        value = float(values[item_idx - 1])
        for cap in range(capacity + 1):
            dp[item_idx, cap] = dp[item_idx - 1, cap]
            if weight <= cap:
                take_value = dp[item_idx - 1, cap - weight] + value
                if take_value > dp[item_idx, cap]:
                    dp[item_idx, cap] = take_value
                    keep[item_idx, cap] = 1

    selected = [0 for _ in range(n_items)]
    cap = capacity
    for item_idx in range(n_items, 0, -1):
        if keep[item_idx, cap] == 1:
            selected[item_idx - 1] = 1
            cap -= int(weights[item_idx - 1])
    return selected


def _fallback_summary(frame_scores: np.ndarray, budget_ratio: float) -> np.ndarray:
    summary = np.zeros((len(frame_scores),), dtype=np.int32)
    if summary.size == 0:
        return summary

    threshold = float(np.quantile(frame_scores, 1 - budget_ratio)) if budget_ratio < 1 else float(np.min(frame_scores))
    summary[frame_scores > threshold] = 1
    if np.any(summary):
        return summary

    sorted_indices = np.argsort(frame_scores)[::-1]
    target_ratio = max(0.0, float(budget_ratio))
    cursor = 0
    while summary.mean() < target_ratio and cursor < len(sorted_indices):
        summary[int(sorted_indices[cursor])] = 1
        cursor += 1
    return summary


def _scene_ranges_to_segments(scene_ranges: Sequence[dict | tuple[int, int]], n_frames: int) -> list[tuple[int, int]]:
    if not scene_ranges:
        return [(0, n_frames)] if n_frames > 0 else []

    segments: list[tuple[int, int]] = []
    for item in scene_ranges:
        if isinstance(item, dict):
            start = int(item.get("start_frame", 0))
            end = int(item.get("end_frame", 0))
        else:
            start, end = int(item[0]), int(item[1])
        segments.append((start, end))
    return _sanitize_segments(segments, n_frames)


def _sanitize_segments(segments: Sequence[tuple[int, int]], n_frames: int) -> list[tuple[int, int]]:
    sanitized: list[tuple[int, int]] = []
    for start, end in segments:
        safe_start = max(0, min(int(start), n_frames))
        safe_end = max(safe_start, min(int(end), n_frames))
        if safe_end > safe_start:
            sanitized.append((safe_start, safe_end))
    return sanitized


def _binary_prf(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    prediction = _resize_binary(prediction, len(target))
    target = _resize_binary(target, len(prediction))
    true_positive = float(np.sum(prediction * target))
    predicted_positive = float(np.sum(prediction))
    target_positive = float(np.sum(target))

    precision = true_positive / predicted_positive if predicted_positive > 0 else 0.0
    recall = true_positive / target_positive if target_positive > 0 else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def _normalize_user_summary(user_summary: np.ndarray) -> np.ndarray:
    summary = np.asarray(user_summary, dtype=np.float32)
    if summary.ndim != 2:
        raise ValueError(f"Expected 2D user_summary, got shape {summary.shape}.")
    return (summary > 0).astype(np.int32)


def _normalize_user_scores(user_scores: np.ndarray, n_frames: int) -> np.ndarray:
    scores = np.asarray(user_scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"Expected 2D user_scores, got shape {scores.shape}.")
    if scores.shape[0] != n_frames:
        resized = [_resize_vector(scores[:, idx], n_frames) for idx in range(scores.shape[1])]
        scores = np.stack(resized, axis=1)
    return scores


def _resize_binary(values: Iterable[int | float], target_length: int) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.int32)
    array = (array > 0).astype(np.int32)
    if array.shape[0] >= target_length:
        return array[:target_length]
    if array.shape[0] == 0:
        return np.zeros((target_length,), dtype=np.int32)
    padding = np.zeros((target_length - array.shape[0],), dtype=np.int32)
    return np.concatenate([array, padding], axis=0)


def _resize_vector(values: np.ndarray, target_length: int) -> np.ndarray:
    if values.shape[0] >= target_length:
        return values[:target_length]
    if values.shape[0] == 0:
        return np.zeros((target_length,), dtype=np.float32)
    padding = np.repeat(values[-1], target_length - values.shape[0]).astype(np.float32)
    return np.concatenate([values.astype(np.float32), padding], axis=0)
