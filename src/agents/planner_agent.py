from __future__ import annotations

from src.data.schemas import AgentScore, PlannerPlan
from src.llm.client import LLMClient
from src.llm.parser import clamp_score, normalize_weights
from src.llm.prompts import (
    PLANNER_SYSTEM_PROMPT,
    build_planner_plan_user_prompt,
    build_planner_segment_user_prompt,
)


class PlannerAgent:
    EXPERT_KEYS = ["story_agent", "visual_agent", "emotion_agent", "information_agent"]

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def plan_video(self, segment_captions: list[str]) -> PlannerPlan:
        payload = self.llm_client.generate_json(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=build_planner_plan_user_prompt(segment_captions),
        )
        return PlannerPlan(
            video_theme=str(payload.get("video_theme", "general")).strip() or "general",
            global_summary=str(payload.get("global_summary", "")).strip(),
            expert_weights=normalize_weights(payload.get("expert_weights", {}), self.EXPERT_KEYS),
            reason=str(payload.get("reason", "")).strip(),
        )

    def score_segment(
        self,
        planner_plan: PlannerPlan,
        current_caption: str,
        memory_context: str,
    ) -> AgentScore:
        payload = self.llm_client.generate_json(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=build_planner_segment_user_prompt(
                video_theme=planner_plan.video_theme,
                global_summary=planner_plan.global_summary,
                current_caption=current_caption,
                memory_context=memory_context,
            ),
        )
        return AgentScore(
            score=clamp_score(payload.get("score", 0.0)),
            reason=str(payload.get("reason", "")).strip(),
        )