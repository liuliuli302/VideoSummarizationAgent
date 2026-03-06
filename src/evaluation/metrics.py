from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Sequence


LABEL_TO_SCORE = {
    "drop": 0.0,
    "low": 1.0,
    "medium": 2.0,
    "high": 3.0,
}


def normalize_pred_scores(frame_scores: Sequence[str | int | float]) -> List[float]:
    normalized = []
    for item in frame_scores:
        if isinstance(item, str):
            normalized.append(float(LABEL_TO_SCORE.get(item, 0.0)))
        else:
            normalized.append(float(item))
    return normalized


def binarize_scores(scores: Sequence[str | int | float], threshold: float | None = None) -> List[int]:
    numeric_scores = normalize_pred_scores(scores)
    if not numeric_scores:
        return []

    effective_threshold = threshold if threshold is not None else _default_threshold(numeric_scores)
    return [1 if score >= effective_threshold else 0 for score in numeric_scores]


def precision_recall_fscore(
    predicted_scores: Sequence[str | int | float],
    gt_scores: Sequence[int | float],
    threshold: float | None = None,
) -> dict:
    _validate_equal_length(predicted_scores, gt_scores)
    predicted_binary = binarize_scores(predicted_scores, threshold=threshold)
    gt_binary = [1 if float(score) > 0 else 0 for score in gt_scores]

    true_positive = sum(1 for pred, gt in zip(predicted_binary, gt_binary) if pred == 1 and gt == 1)
    false_positive = sum(1 for pred, gt in zip(predicted_binary, gt_binary) if pred == 1 and gt == 0)
    false_negative = sum(1 for pred, gt in zip(predicted_binary, gt_binary) if pred == 0 and gt == 1)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    if precision + recall == 0:
        fscore = 0.0
    else:
        fscore = 2 * precision * recall / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "fscore": round(fscore, 4),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
    }


def spearman_correlation(
    predicted_scores: Sequence[str | int | float],
    gt_scores: Sequence[int | float],
) -> float:
    _validate_equal_length(predicted_scores, gt_scores)
    pred_ranks = _rankdata(normalize_pred_scores(predicted_scores))
    gt_ranks = _rankdata([float(item) for item in gt_scores])
    return round(_pearson(pred_ranks, gt_ranks), 4)


def kendall_correlation(
    predicted_scores: Sequence[str | int | float],
    gt_scores: Sequence[int | float],
) -> float:
    _validate_equal_length(predicted_scores, gt_scores)
    pred_values = normalize_pred_scores(predicted_scores)
    gt_values = [float(item) for item in gt_scores]
    n = len(pred_values)
    if n < 2:
        return 0.0

    concordant = 0
    discordant = 0
    for i, j in combinations(range(n), 2):
        pred_diff = pred_values[i] - pred_values[j]
        gt_diff = gt_values[i] - gt_values[j]
        product = pred_diff * gt_diff
        if product > 0:
            concordant += 1
        elif product < 0:
            discordant += 1

    total_pairs = n * (n - 1) / 2
    if total_pairs == 0:
        return 0.0
    return round((concordant - discordant) / total_pairs, 4)


def coverage_score(
    predicted_scores: Sequence[str | int | float],
    gt_scores: Sequence[int | float],
    threshold: float | None = None,
) -> float:
    _validate_equal_length(predicted_scores, gt_scores)
    predicted_binary = binarize_scores(predicted_scores, threshold=threshold)
    gt_binary = [1 if float(score) > 0 else 0 for score in gt_scores]
    positives = sum(gt_binary)
    if positives == 0:
        return 0.0
    covered = sum(1 for pred, gt in zip(predicted_binary, gt_binary) if pred == 1 and gt == 1)
    return round(covered / positives, 4)


def diversity_score(selected_segments: Sequence[dict]) -> float:
    if len(selected_segments) <= 1:
        return 1.0 if selected_segments else 0.0

    pairwise_scores = []
    for left, right in combinations(selected_segments, 2):
        left_text = _segment_text(left)
        right_text = _segment_text(right)
        text_diversity = 1.0 - _jaccard_similarity(left_text.split(), right_text.split())

        left_center = (int(left.get("start_frame", 0)) + int(left.get("end_frame", 0))) / 2.0
        right_center = (int(right.get("start_frame", 0)) + int(right.get("end_frame", 0))) / 2.0
        max_frame = max(int(left.get("end_frame", 0)), int(right.get("end_frame", 0)), 1)
        temporal_diversity = min(abs(left_center - right_center) / max_frame, 1.0)

        pairwise_scores.append((text_diversity + temporal_diversity) / 2.0)

    return round(sum(pairwise_scores) / len(pairwise_scores), 4)


def latency_statistics(latencies_sec: Iterable[float]) -> dict:
    values = [float(item) for item in latencies_sec]
    if not values:
        return {"count": 0, "mean_sec": 0.0, "max_sec": 0.0, "min_sec": 0.0, "p95_sec": 0.0}

    ordered = sorted(values)
    count = len(ordered)
    mean = sum(ordered) / count
    p95_index = min(count - 1, max(0, int(round(0.95 * (count - 1)))))
    return {
        "count": count,
        "mean_sec": round(mean, 4),
        "max_sec": round(ordered[-1], 4),
        "min_sec": round(ordered[0], 4),
        "p95_sec": round(ordered[p95_index], 4),
    }


def _default_threshold(scores: Sequence[float]) -> float:
    max_score = max(scores) if scores else 0.0
    if max_score >= 2.0:
        return 2.0
    if max_score >= 1.0:
        return 1.0
    return 0.0


def _validate_equal_length(left: Sequence[object], right: Sequence[object]) -> None:
    if len(left) != len(right):
        raise ValueError(f"Input lengths must match, got {len(left)} and {len(right)}.")


def _rankdata(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[start][1]:
            end += 1
        average_rank = (start + end + 2) / 2.0
        for offset in range(start, end + 1):
            original_index = indexed[offset][0]
            ranks[original_index] = average_rank
        start = end + 1
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Pearson inputs must have equal length.")
    if not left:
        return 0.0

    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_denom = sum((x - left_mean) ** 2 for x in left)
    right_denom = sum((y - right_mean) ** 2 for y in right)
    denominator = (left_denom * right_denom) ** 0.5
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _jaccard_similarity(left_tokens: Sequence[str], right_tokens: Sequence[str]) -> float:
    left_set = {token.lower() for token in left_tokens if token}
    right_set = {token.lower() for token in right_tokens if token}
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 0.0
    intersection = left_set & right_set
    return len(intersection) / len(union)


def _segment_text(segment: dict) -> str:
    return str(segment.get("summary_text") or segment.get("selection_reason") or "")