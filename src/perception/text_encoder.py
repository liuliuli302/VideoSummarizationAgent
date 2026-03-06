from __future__ import annotations

from typing import Iterable, Optional


class WindowTextEncoder:
    """Build a semantic summary from local caption and optional text cues."""

    def build_semantic_summary(
        self,
        local_caption: str,
        title: Optional[str] = None,
        asr_text: Optional[str | Iterable[dict]] = None,
    ) -> str:
        normalized_caption = self._normalize_sentence(local_caption)
        if not normalized_caption:
            raise ValueError("local_caption must not be empty.")

        parts = [f"Local event: {normalized_caption}"]

        normalized_title = self._normalize_sentence(title)
        if normalized_title:
            parts.append(f"Title context: {normalized_title}")

        normalized_asr = self._normalize_asr(asr_text)
        if normalized_asr:
            parts.append(f"ASR cue: {normalized_asr}")

        return " ".join(parts)

    def _normalize_asr(self, asr_text: Optional[str | Iterable[dict]]) -> str:
        if asr_text is None:
            return ""

        if isinstance(asr_text, str):
            return self._normalize_sentence(asr_text)

        extracted_segments = []
        for segment in asr_text:
            if not isinstance(segment, dict):
                continue
            content = segment.get("text") or segment.get("utterance") or segment.get("content")
            normalized = self._normalize_sentence(content)
            if normalized:
                extracted_segments.append(normalized)

        return " ".join(extracted_segments)

    def _normalize_sentence(self, text: Optional[str]) -> str:
        if text is None:
            return ""

        normalized = " ".join(str(text).strip().split())
        if not normalized:
            return ""
        if normalized.endswith((".", "!", "?", "。", "！", "？")):
            return normalized
        return f"{normalized}."