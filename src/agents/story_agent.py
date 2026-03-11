from __future__ import annotations

from src.agents.base_agent import BaseExpertAgent
from src.llm.client import LLMClient


class StoryAgent(BaseExpertAgent):
    def __init__(self, llm_client: LLMClient) -> None:
        super().__init__(
            llm_client=llm_client,
            agent_name="story_agent",
            dimension_desc="how much this segment contributes to story progression or main event progression",
        )