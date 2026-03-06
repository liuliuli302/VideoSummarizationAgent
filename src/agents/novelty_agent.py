from __future__ import annotations

from src.agents._text_utils import best_overlap


class NoveltyAgent:
    """Baseline rule-based novelty judge."""

    def run(self, window_summary: str, selected_memory: list[str], recent_context: list[str]) -> str:
        selected_overlap = best_overlap(window_summary, selected_memory)
        recent_overlap = best_overlap(window_summary, recent_context)
        max_overlap = max(selected_overlap, recent_overlap)

        if max_overlap < 0.15:
            novelty = "高"
            repetition = "否"
            reason = "当前窗口与已选内容和最近上下文差异明显，提供了新的信息。"
        elif max_overlap < 0.32:
            novelty = "中"
            repetition = "部分重复"
            reason = "当前窗口与历史内容存在局部重叠，但仍补充了部分新信息。"
        else:
            novelty = "低"
            repetition = "是"
            reason = "当前窗口与已保留内容高度相似，新增信息有限。"

        return (
            f"Novelty: {novelty}\n"
            f"Repetition: {repetition}\n"
            f"Reason: {reason}"
        )