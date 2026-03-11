from __future__ import annotations

from src.data.schemas import ExpertResult
from src.llm.client import LLMClient
from src.llm.parser import clamp_score
from src.llm.prompts import EXPERT_SYSTEM_PROMPT_TEMPLATE, build_expert_user_prompt


class BaseExpertAgent:
    def __init__(self, llm_client: LLMClient, agent_name: str, dimension_desc: str) -> None:
        self.llm_client = llm_client
        self.agent_name = agent_name
        self.dimension_desc = dimension_desc

    def score_segment(
        self,
        video_theme: str,
        global_summary: str,
        current_caption: str,
        memory_context: str,
    ) -> ExpertResult:
        payload = self.llm_client.generate_json(
            system_prompt=EXPERT_SYSTEM_PROMPT_TEMPLATE.format(agent_name=self.agent_name),
            user_prompt=build_expert_user_prompt(
                dimension_desc=self.dimension_desc,
                video_theme=video_theme,
                global_summary=global_summary,
                current_caption=current_caption,
                memory_context=memory_context,
            ),
        )
        return ExpertResult(
            agent_name=self.agent_name,
            score=clamp_score(payload.get("score", 0.0)),
            reason=str(payload.get("reason", "")).strip(),
        )