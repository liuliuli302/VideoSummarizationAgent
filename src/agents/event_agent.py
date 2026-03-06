from __future__ import annotations

from src.agents._text_utils import contains_any


class EventAgent:
    """Baseline rule-based event importance judge."""

    def run(self, visual_description: str, local_caption: str, asr_text: str | None = None) -> str:
        combined_text = " ".join(part for part in [visual_description, local_caption, asr_text or ""] if part)
        lower_text = combined_text.lower()

        high_keywords = ["dynamic", "strong motion", "rapid", "crowd", "goal", "celebr", "busy"]
        medium_keywords = ["moderate", "arrive", "move", "lesson", "explain", "changing"]

        if contains_any(lower_text, high_keywords):
            importance = "高"
            event_type = self._infer_event_type(lower_text)
            reason = "窗口包含明显动作变化或高密度事件信号，适合进入摘要。"
        elif contains_any(lower_text, medium_keywords):
            importance = "中"
            event_type = self._infer_event_type(lower_text)
            reason = "窗口包含可识别事件，但强度和稀缺性中等。"
        else:
            importance = "低"
            event_type = self._infer_event_type(lower_text)
            reason = "窗口以平稳场景描述为主，缺少显著高光信号。"

        return (
            f"Event Importance: {importance}\n"
            f"Event Type: {event_type}\n"
            f"Reason: {reason}"
        )

    def _infer_event_type(self, text: str) -> str:
        if any(keyword in text for keyword in ["goal", "match", "team", "race", "sports"]):
            return "sports_event"
        if any(keyword in text for keyword in ["lesson", "explain", "lecture", "class"]):
            return "instruction"
        if any(keyword in text for keyword in ["travel", "city", "street", "downtown", "journey"]):
            return "travel_activity"
        if any(keyword in text for keyword in ["crowd", "move", "traffic", "busy"]):
            return "activity_peak"
        return "scene_update"