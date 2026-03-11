from __future__ import annotations

from src.agents.base_agent import BaseExpertAgent
from src.llm.client import LLMClient


class EmotionAgent(BaseExpertAgent):
    def __init__(self, llm_client: LLMClient) -> None:
        super().__init__(
            llm_client=llm_client,
            agent_name="emotion_agent",
            dimension_desc="how strong the emotional expression, tension, or sentiment is in this segment",
        )