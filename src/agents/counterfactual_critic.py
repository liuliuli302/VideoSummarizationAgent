from __future__ import annotations

from typing import Iterable

from src.agents._text_utils import best_overlap, contains_any, overlap_ratio
from src.datasets.schemas import MemoryState


class CounterfactualCritic:
    """Baseline rule-based critic for marginal contribution estimation."""

    def run(
        self,
        memory_summary: MemoryState | str | Iterable[str],
        window_summary: str,
        summary_goal: str,
    ) -> str:
        memory_texts = self._normalize_memory(memory_summary)
        memory_overlap = best_overlap(window_summary, memory_texts)
        goal_overlap = overlap_ratio(window_summary, summary_goal)
        event_signal = contains_any(
            window_summary,
            ["dynamic", "busy", "crowd", "goal", "travel", "lesson", "match", "downtown", "critical"],
        )
        theme_signal = contains_any(summary_goal, ["travel", "sports", "lesson", "story", "city"])

        if (goal_overlap >= 0.16 and memory_overlap < 0.18) or (theme_signal and event_signal and memory_overlap < 0.12):
            contribution = "高"
            loss_if_removed = "删除后会损失与全局目标直接相关的新信息，摘要主线会变得不完整。"
            recommendation = "保留"
        elif event_signal and memory_overlap < 0.28:
            contribution = "中"
            loss_if_removed = "删除后会弱化关键事件或场景变化的可见性，但仍可由其他片段部分补充。"
            recommendation = "视预算决定"
        elif memory_overlap >= 0.28:
            contribution = "低"
            loss_if_removed = "删除后损失有限，因为当前窗口与已有摘要记忆存在较强重叠。"
            recommendation = "可省略"
        else:
            contribution = "中"
            loss_if_removed = "删除后会减少部分上下文信息，但不会明显破坏整体理解。"
            recommendation = "视预算决定"

        return (
            f"Marginal Contribution: {contribution}\n"
            f"Loss If Removed: {loss_if_removed}\n"
            f"Recommendation: {recommendation}"
        )

    def _normalize_memory(self, memory_summary: MemoryState | str | Iterable[str]) -> list[str]:
        if isinstance(memory_summary, MemoryState):
            return [
                *memory_summary.story_slots,
                *memory_summary.selected_slots,
                *memory_summary.temporal_context,
            ]
        if isinstance(memory_summary, str):
            return [memory_summary]
        return [str(item) for item in memory_summary]