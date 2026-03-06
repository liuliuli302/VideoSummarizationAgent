from __future__ import annotations

import re
from typing import Dict, Set


class ConsistencyChecker:
    """Heuristic consistency checker for visual and caption texts."""

    def check(self, visual_description: str, local_caption: str) -> str:
        visual_tokens = self._normalize_tokens(visual_description)
        caption_tokens = self._normalize_tokens(local_caption)

        if not visual_tokens or not caption_tokens:
            return "Consistency: low; Reason: missing visual or caption content."

        overlap = visual_tokens & caption_tokens
        overlap_ratio = len(overlap) / max(1, min(len(visual_tokens), len(caption_tokens)))

        if overlap_ratio >= 0.35:
            level = "high"
            reason = "visual description and local caption share key scene words"
        elif overlap_ratio >= 0.15:
            level = "medium"
            reason = "visual description and local caption partially align"
        else:
            level = "low"
            reason = "visual description and local caption emphasize different details"

        return f"Consistency: {level}; Reason: {reason}."

    def check_with_details(self, visual_description: str, local_caption: str) -> Dict[str, object]:
        visual_tokens = self._normalize_tokens(visual_description)
        caption_tokens = self._normalize_tokens(local_caption)
        overlap = visual_tokens & caption_tokens
        overlap_ratio = len(overlap) / max(1, min(len(visual_tokens), len(caption_tokens))) if visual_tokens and caption_tokens else 0.0

        comment = self.check(visual_description, local_caption)
        level = comment.split(";", maxsplit=1)[0].split(":", maxsplit=1)[1].strip()
        return {
            "level": level,
            "comment": comment,
            "shared_tokens": sorted(overlap),
            "overlap_ratio": overlap_ratio,
        }

    def _normalize_tokens(self, text: str) -> Set[str]:
        raw_tokens = re.findall(r"[a-zA-Z]+", text.lower())
        stop_words = {
            "the", "a", "an", "and", "or", "with", "from", "into", "window", "sampled",
            "frames", "scene", "motion", "visual", "change", "brightness", "dominant", "color",
        }
        return {token for token in raw_tokens if len(token) > 2 and token not in stop_words}