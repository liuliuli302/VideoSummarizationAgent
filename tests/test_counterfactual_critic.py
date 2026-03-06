import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents import CounterfactualCritic
from src.datasets.schemas import MemoryState


class TestCounterfactualCriticStep10(unittest.TestCase):
    def setUp(self):
        self.critic = CounterfactualCritic()

    def test_high_marginal_contribution(self):
        memory = MemoryState(
            story_slots=["Local event: arrival at the city gate."],
            selected_slots=["Local event: entering downtown streets."],
            redundancy_bank=[],
            temporal_context=["Local event: walking toward the central plaza."],
        )
        result = self.critic.run(
            memory_summary=memory,
            window_summary="Local event: the traveler reaches a busy downtown square with crowd activity.",
            summary_goal="Primary theme: travel. Summary goal: preserve the main progression and representative city activities.",
        )

        self.assertIn("Marginal Contribution:", result)
        self.assertIn("Recommendation:", result)
        self.assertIn("保留", result)

    def test_low_marginal_contribution_when_redundant(self):
        memory = MemoryState(
            story_slots=["Local event: a crowded downtown street with traffic and pedestrians."],
            selected_slots=["Local event: a crowded downtown street with traffic and pedestrians."],
            redundancy_bank=[],
            temporal_context=[],
        )
        result = self.critic.run(
            memory_summary=memory,
            window_summary="Local event: a crowded downtown street with traffic and pedestrians.",
            summary_goal="Primary theme: travel.",
        )

        self.assertIn("Marginal Contribution: 低", result)
        self.assertIn("Recommendation: 可省略", result)


if __name__ == "__main__":
    unittest.main()