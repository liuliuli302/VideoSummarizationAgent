from __future__ import annotations

from src.agents._text_utils import contains_any


class DomainAgent:
    """Baseline rule-based domain preference judge."""

    def run(self, video_theme: str, domain_hint: str, window_summary: str) -> str:
        combined = " ".join([video_theme, domain_hint, window_summary]).lower()

        if contains_any(combined, ["travel", "city", "street", "journey", "vlog"]):
            match = "高"
            policy = "优先保留能体现地点变化和行程推进的窗口。"
            reason = "当前窗口符合旅行类摘要偏好，能够体现地点和活动信息。"
        elif contains_any(combined, ["lesson", "tutorial", "lecture", "class", "explain"]):
            match = "高"
            policy = "优先保留知识点讲解和关键演示窗口。"
            reason = "当前窗口符合教学类摘要偏好，能够体现解释或演示内容。"
        elif contains_any(combined, ["sport", "goal", "match", "team", "race"]):
            match = "高"
            policy = "优先保留对抗、高光和比分变化窗口。"
            reason = "当前窗口符合体育类摘要偏好，具有明显赛事信息。"
        elif contains_any(combined, ["story", "character", "dialogue", "scene", "plot"]):
            match = "中"
            policy = "保留推动剧情或人物关系变化的窗口。"
            reason = "当前窗口与叙事型内容有关，但领域价值仍需结合主线判断。"
        else:
            match = "低"
            policy = "仅在预算充足时保留该类通用窗口。"
            reason = "当前窗口缺少明确领域偏好信号，领域加成较弱。"

        return (
            f"Domain Match: {match}\n"
            f"Policy Suggestion: {policy}\n"
            f"Reason: {reason}"
        )