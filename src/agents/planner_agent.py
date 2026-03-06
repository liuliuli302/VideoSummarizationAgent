from __future__ import annotations

from typing import List

from src.agents._text_utils import best_overlap, contains_any, overlap_ratio
from src.datasets.schemas import MemoryState, PlannerOutput


class PlannerAgent:
    """Baseline planner that emits text routing plans without numeric weights."""

    def run(
        self,
        window_context_text: str,
        summary_goal_text: str,
        memory_state: MemoryState,
    ) -> PlannerOutput:
        goal_overlap = overlap_ratio(window_context_text, summary_goal_text)
        selected_overlap = best_overlap(window_context_text, memory_state.selected_slots)
        story_overlap = best_overlap(window_context_text, memory_state.story_slots)
        temporal_overlap = best_overlap(window_context_text, memory_state.temporal_context[-3:])

        focus_points = self._build_focus_points(
            window_context_text=window_context_text,
            goal_overlap=goal_overlap,
            selected_overlap=selected_overlap,
            story_overlap=story_overlap,
            temporal_overlap=temporal_overlap,
        )

        priority_order = self._build_priority_order(
            window_context_text=window_context_text,
            goal_overlap=goal_overlap,
            selected_overlap=selected_overlap,
            temporal_overlap=temporal_overlap,
        )

        route_plan_text = self._format_route_plan(priority_order=priority_order, focus_points=focus_points)
        routing_rationale = self._build_rationale(
            goal_overlap=goal_overlap,
            selected_overlap=selected_overlap,
            story_overlap=story_overlap,
            temporal_overlap=temporal_overlap,
        )

        return PlannerOutput(
            route_plan_text=route_plan_text,
            focus_points=focus_points,
            routing_rationale=routing_rationale,
        )

    def _build_focus_points(
        self,
        window_context_text: str,
        goal_overlap: float,
        selected_overlap: float,
        story_overlap: float,
        temporal_overlap: float,
    ) -> List[str]:
        focus_points: List[str] = []

        if goal_overlap >= 0.1 or story_overlap >= 0.1:
            focus_points.append("优先确认是否推动主线")
        if selected_overlap >= 0.12:
            focus_points.append("重点检查与已选摘要的重复")
        if temporal_overlap >= 0.08:
            focus_points.append("检查是否承担前后片段衔接作用")
        if contains_any(window_context_text, ["dynamic", "strong motion", "crowd", "busy", "goal", "rapid"]):
            focus_points.append("关注是否存在高光事件")
        if contains_any(window_context_text, ["travel", "city", "lesson", "sports", "match", "tutorial", "story"]):
            focus_points.append("结合领域偏好判断保留价值")

        if not focus_points:
            focus_points.append("先做通用主线与新颖性检查")

        return focus_points

    def _build_priority_order(
        self,
        window_context_text: str,
        goal_overlap: float,
        selected_overlap: float,
        temporal_overlap: float,
    ) -> List[str]:
        order: List[str] = []

        if goal_overlap >= 0.1:
            order.append("MainlineAgent")
        if selected_overlap >= 0.12:
            order.append("NoveltyAgent")
        if contains_any(window_context_text, ["dynamic", "strong motion", "crowd", "busy", "goal", "rapid"]):
            order.append("EventAgent")
        if temporal_overlap >= 0.08:
            order.append("TemporalAgent")
        if contains_any(window_context_text, ["travel", "city", "lesson", "sports", "match", "tutorial", "story"]):
            order.append("DomainAgent")

        if "MainlineAgent" not in order:
            order.insert(0, "MainlineAgent")
        if "NoveltyAgent" not in order:
            insert_pos = 1 if len(order) >= 1 else 0
            order.insert(insert_pos, "NoveltyAgent")
        if "EventAgent" not in order:
            order.append("EventAgent")

        deduplicated = []
        seen = set()
        for name in order:
            if name not in seen:
                deduplicated.append(name)
                seen.add(name)
        return deduplicated

    def _format_route_plan(self, priority_order: List[str], focus_points: List[str]) -> str:
        expert_order = " -> ".join(priority_order)
        focus_text = "；".join(focus_points)
        return f"Priority Experts: {expert_order}\nFocus Points: {focus_text}"

    def _build_rationale(
        self,
        goal_overlap: float,
        selected_overlap: float,
        story_overlap: float,
        temporal_overlap: float,
    ) -> str:
        reasons = []
        if goal_overlap >= 0.1:
            reasons.append("当前窗口与全局摘要目标相关，需优先检查主线推进")
        if story_overlap >= 0.1:
            reasons.append("当前窗口与主线记忆接近，需要确认是推进还是重复")
        if selected_overlap >= 0.12:
            reasons.append("当前窗口与已选摘要存在重叠，需先做新颖性筛查")
        if temporal_overlap >= 0.08:
            reasons.append("当前窗口与最近时序上下文接近，可能承担过渡作用")
        if not reasons:
            reasons.append("当前窗口缺少明显偏置信号，采用主线-新颖性-事件的通用审查顺序")
        return "；".join(reasons)