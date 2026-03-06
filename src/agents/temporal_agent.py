from __future__ import annotations

from src.agents._text_utils import overlap_ratio


class TemporalAgent:
    """Baseline rule-based temporal continuity judge."""

    def run(
        self,
        prev_summary: str,
        current_summary: str,
        next_summary: str,
        summary_chain: list[str],
    ) -> str:
        prev_overlap = overlap_ratio(current_summary, prev_summary)
        next_overlap = overlap_ratio(current_summary, next_summary)
        chain_overlap = max((overlap_ratio(current_summary, item) for item in summary_chain), default=0.0)

        if prev_overlap >= 0.08 and next_overlap >= 0.08:
            value = "高"
            role = "承接"
            reason = "当前窗口同时连接前后语义，能增强摘要的连续性。"
        elif prev_overlap >= 0.05 or next_overlap >= 0.05 or chain_overlap >= 0.08:
            value = "中"
            role = "过渡"
            reason = "当前窗口具有一定衔接作用，但不是唯一的连续性支点。"
        else:
            value = "低"
            role = "可省略"
            reason = "当前窗口与上下文衔接较弱，对摘要时序帮助有限。"

        return (
            f"Temporal Value: {value}\n"
            f"Transition Role: {role}\n"
            f"Reason: {reason}"
        )