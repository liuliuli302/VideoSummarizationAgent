from __future__ import annotations

from src.agents.base_agent import BaseExpertAgent
from src.llm.client import LLMClient


class VisualAgent(BaseExpertAgent):
    def __init__(self, llm_client: LLMClient) -> None:
        super().__init__(
            llm_client=llm_client,
            agent_name="visual_agent",
            dimension_desc="how visually striking, diverse, or highlight-worthy this segment is",
        )