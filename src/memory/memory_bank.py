from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

from src.agents._text_utils import best_overlap
from src.datasets.schemas import MemoryState, WindowFeature
from src.memory.retrieval import retrieve_relevant_texts


class MemoryBank:
    """Rule-based long-term memory state manager."""

    def __init__(
        self,
        topk: int = 5,
        max_items_per_slot: int = 50,
        similarity_prune_threshold: float = 0.9,
    ):
        if topk <= 0:
            raise ValueError(f"topk must be positive, got {topk}.")
        if max_items_per_slot <= 0:
            raise ValueError(f"max_items_per_slot must be positive, got {max_items_per_slot}.")
        if not 0.0 <= similarity_prune_threshold <= 1.0:
            raise ValueError(
                "similarity_prune_threshold must fall inside [0, 1], "
                f"got {similarity_prune_threshold}."
            )

        self.topk = topk
        self.max_items_per_slot = max_items_per_slot
        self.similarity_prune_threshold = similarity_prune_threshold
        self.state = MemoryState()

    def read(self, current_summary: str, topk: Optional[int] = None) -> Dict[str, list[str]]:
        effective_topk = topk or self.topk
        return {
            "selected_ctx": retrieve_relevant_texts(
                self.state.selected_slots,
                current_summary,
                topk=effective_topk,
            ),
            "story_ctx": retrieve_relevant_texts(
                self.state.story_slots,
                current_summary,
                topk=effective_topk,
            ),
            "temporal_ctx": self.state.temporal_context[-effective_topk:],
        }

    def retrieve_topk(self, slot_name: str, query: str, topk: Optional[int] = None) -> list[str]:
        effective_topk = topk or self.topk
        slot_values = self._get_slot(slot_name)
        return retrieve_relevant_texts(slot_values, query, topk=effective_topk)

    def update(
        self,
        window_feature: WindowFeature,
        final_decision: str,
        is_selected: bool,
    ) -> MemoryState:
        summary_text = window_feature.semantic_summary
        self.state.temporal_context.append(summary_text)
        self._append_unique(self.state.redundancy_bank, summary_text)

        if is_selected:
            self._append_unique(self.state.selected_slots, summary_text)

        if final_decision in {"必须保留", "建议保留"}:
            self._append_unique(self.state.story_slots, summary_text)

        self._trim_state()
        return self.snapshot()

    def snapshot(self) -> MemoryState:
        return replace(
            self.state,
            story_slots=list(self.state.story_slots),
            selected_slots=list(self.state.selected_slots),
            redundancy_bank=list(self.state.redundancy_bank),
            temporal_context=list(self.state.temporal_context),
        )

    def reset(self) -> None:
        self.state = MemoryState()

    def _get_slot(self, slot_name: str) -> list[str]:
        if not hasattr(self.state, slot_name):
            raise ValueError(f"Unknown memory slot: {slot_name}")
        slot_value = getattr(self.state, slot_name)
        if not isinstance(slot_value, list):
            raise ValueError(f"Memory slot {slot_name} is not a list.")
        return slot_value

    def _append_unique(self, target: list[str], item: str) -> None:
        normalized_item = " ".join(item.strip().split())
        if not normalized_item:
            return
        if normalized_item in target:
            return
        if target and best_overlap(normalized_item, target) >= self.similarity_prune_threshold:
            return
        target.append(normalized_item)

    def _trim_state(self) -> None:
        self.state.story_slots = self.state.story_slots[-self.max_items_per_slot :]
        self.state.selected_slots = self.state.selected_slots[-self.max_items_per_slot :]
        self.state.redundancy_bank = self.state.redundancy_bank[-self.max_items_per_slot :]
        self.state.temporal_context = self.state.temporal_context[-self.max_items_per_slot :]