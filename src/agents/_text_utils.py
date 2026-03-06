from __future__ import annotations

import re
from typing import Iterable, Set


def tokenize_text(text: str | Iterable[str]) -> Set[str]:
    if isinstance(text, str):
        raw = text.lower()
    else:
        raw = " ".join(str(item) for item in text).lower()

    tokens = re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", raw)
    stop_words = {
        "the", "and", "with", "for", "from", "that", "this", "window", "segment",
        "visual", "description", "local", "caption", "semantic", "summary", "reason",
        "event", "context", "current", "video", "scene",
    }
    return {token for token in tokens if len(token) > 1 and token not in stop_words}


def overlap_ratio(text_a: str | Iterable[str], text_b: str | Iterable[str]) -> float:
    tokens_a = tokenize_text(text_a)
    tokens_b = tokenize_text(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / max(1, len(union))


def best_overlap(query: str, candidates: Iterable[str]) -> float:
    return max((overlap_ratio(query, candidate) for candidate in candidates), default=0.0)


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    normalized = text.lower()
    return any(keyword.lower() in normalized for keyword in keywords)