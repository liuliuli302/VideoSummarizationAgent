from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple


def retrieve_relevant_texts(
    texts: Sequence[str],
    query: str,
    topk: int = 5,
) -> List[str]:
    """Return top-k texts ranked by simple lexical overlap with the query."""
    if topk <= 0 or not texts:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return list(texts[:topk])

    scored_texts: List[Tuple[float, int, str]] = []
    for index, text in enumerate(texts):
        score = _overlap_score(query_tokens, _tokenize(text))
        scored_texts.append((score, index, text))

    scored_texts.sort(key=lambda item: (-item[0], item[1]))
    ranked = [text for score, _, text in scored_texts if score > 0]
    if ranked:
        return ranked[:topk]

    return list(texts[:topk])


def _overlap_score(query_tokens: set[str], text_tokens: set[str]) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    intersection = query_tokens & text_tokens
    union = query_tokens | text_tokens
    weighted_intersection = sum(len(token) for token in intersection)
    weighted_query = sum(len(token) for token in query_tokens)
    weighted_recall = weighted_intersection / max(1, weighted_query)
    jaccard = len(intersection) / max(1, len(union))
    return weighted_recall + (0.1 * jaccard)


def _tokenize(text: str | Iterable[str]) -> set[str]:
    if isinstance(text, str):
        raw = text.lower()
    else:
        raw = " ".join(str(item) for item in text).lower()

    tokens = re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", raw)
    stop_words = {
        "the", "and", "with", "for", "from", "that", "this", "window", "segment",
        "visual", "description", "local", "caption", "semantic", "summary", "scene",
    }
    return {token for token in tokens if len(token) > 1 and token not in stop_words}