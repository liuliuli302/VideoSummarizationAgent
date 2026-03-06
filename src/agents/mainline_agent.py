from __future__ import annotations

from src.agents._text_utils import best_overlap, overlap_ratio


class MainlineAgent:
    """Baseline rule-based mainline progression judge."""

    def run(self, window_summary: str, summary_goal: str, story_memory: list[str]) -> str:
        goal_overlap = overlap_ratio(window_summary, summary_goal)
        memory_overlap = best_overlap(window_summary, story_memory)

        if goal_overlap >= 0.18 and memory_overlap < 0.2:
            judgment = "强"
            conclusion = "当前窗口为全局目标提供了新的主线推进。"
            reason = "它与摘要目标高度相关，同时与已有主线记忆重复较少。"
        elif goal_overlap >= 0.08:
            judgment = "中"
            conclusion = "当前窗口与主线有关，但推进强度有限。"
            reason = "它与摘要目标存在一定关联，不过部分内容与已有主线记忆接近。"
        else:
            judgment = "弱"
            conclusion = "当前窗口对主线推进贡献较弱。"
            reason = "它与全局摘要目标关联较低，缺少明确的新主线信息。"

        return (
            f"Mainline Judgment: {judgment}\n"
            f"Conclusion: {conclusion}\n"
            f"Reason: {reason}"
        )