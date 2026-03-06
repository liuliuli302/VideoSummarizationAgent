from __future__ import annotations

from typing import Iterable, Optional

from src.datasets.schemas import Window, WindowFeature
from src.perception.consistency import ConsistencyChecker


class WindowFeatureBuilder:
    """Build ``WindowFeature`` objects from text-side local perception signals."""

    def __init__(self, consistency_checker: Optional[ConsistencyChecker] = None):
        self.consistency_checker = consistency_checker or ConsistencyChecker()

    def build(
        self,
        window: Window,
        visual_description: str,
        local_caption: str,
        semantic_summary: str,
        asr_text: Optional[str | Iterable[dict]] = None,
    ) -> WindowFeature:
        consistency = self.consistency_checker.check_with_details(
            visual_description=visual_description,
            local_caption=local_caption,
        )
        evidence_notes = self._build_evidence_notes(
            consistency_level=str(consistency["level"]),
            consistency_comment=str(consistency["comment"]),
            shared_tokens=list(consistency["shared_tokens"]),
            asr_text=asr_text,
        )

        enriched_summary = self._build_context_text(
            visual_description=visual_description,
            local_caption=local_caption,
            semantic_summary=semantic_summary,
            asr_text=asr_text,
            consistency_comment=str(consistency["comment"]),
        )

        return WindowFeature(
            win_id=window.win_id,
            visual_description=visual_description,
            local_caption=local_caption,
            semantic_summary=enriched_summary,
            evidence_notes=evidence_notes,
        )

    def _build_context_text(
        self,
        visual_description: str,
        local_caption: str,
        semantic_summary: str,
        asr_text: Optional[str | Iterable[dict]],
        consistency_comment: str,
    ) -> str:
        parts = [
            f"Visual Description: {visual_description}",
            f"Local Caption: {local_caption}",
            f"Semantic Summary: {semantic_summary}",
            consistency_comment,
        ]

        normalized_asr = self._normalize_asr(asr_text)
        if normalized_asr:
            parts.append(f"ASR: {normalized_asr}")

        return "\n".join(parts)

    def _build_evidence_notes(
        self,
        consistency_level: str,
        consistency_comment: str,
        shared_tokens: list[str],
        asr_text: Optional[str | Iterable[dict]],
    ) -> list[str]:
        notes = [f"consistency_level={consistency_level}", consistency_comment]
        if shared_tokens:
            notes.append(f"shared_tokens={', '.join(shared_tokens)}")

        normalized_asr = self._normalize_asr(asr_text)
        if normalized_asr:
            notes.append(f"asr_present={normalized_asr}")

        return notes

    def _normalize_asr(self, asr_text: Optional[str | Iterable[dict]]) -> str:
        if asr_text is None:
            return ""

        if isinstance(asr_text, str):
            return " ".join(asr_text.strip().split())

        parts = []
        for item in asr_text:
            if not isinstance(item, dict):
                continue
            text = item.get("text") or item.get("utterance") or item.get("content")
            if text:
                parts.append(" ".join(str(text).strip().split()))
        return " ".join(parts)