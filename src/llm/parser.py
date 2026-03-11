from __future__ import annotations

from typing import Any


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        score = 0.0
    return max(0.0, min(1.0, score))


def normalize_weights(weights: dict[str, Any], keys: list[str]) -> dict[str, float]:
    cleaned = {key: max(0.0, float(weights.get(key, 0.0))) for key in keys}
    total = sum(cleaned.values())
    if total <= 0:
        uniform = 1.0 / len(keys)
        return {key: uniform for key in keys}
    return {key: value / total for key, value in cleaned.items()}