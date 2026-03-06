import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents import PlannerAgent
from src.datasets.schemas import MemoryState


class TestPlannerAgentStep9(unittest.TestCase):
    def setUp(self):
        self.agent = PlannerAgent()

    def test_planner_generates_route_plan(self):
        memory_state = MemoryState(
            story_slots=["Local event: arrival at a busy city center."],
            selected_slots=["Local event: a crowded downtown street with traffic."],
            redundancy_bank=[],
            temporal_context=[
                "Local event: entering the downtown area.",
                "Local event: pedestrians gather at a crossing.",
            ],
        )
        window_context = (
            "Visual Description: bright city street with dynamic crowd motion.\n"
            "Local Caption: pedestrians move through a busy downtown street.\n"
            "Semantic Summary: Local event: pedestrians move through a busy downtown street."
        )
        summary_goal = (
            "Primary theme: travel. Summary goal: preserve the main progression across coarse segments "
            "and retain representative city activities."
        )

        result = self.agent.run(
            window_context_text=window_context,
            summary_goal_text=summary_goal,
            memory_state=memory_state,
        )

        self.assertIn("Priority Experts:", result.route_plan_text)
        self.assertIn("Focus Points:", result.route_plan_text)
        self.assertTrue(result.focus_points)
        self.assertIn("主线", result.routing_rationale)

    def test_planner_fallback_route(self):
        result = self.agent.run(
            window_context_text="Visual Description: static indoor frame.",
            summary_goal_text="Primary theme: general.",
            memory_state=MemoryState(),
        )

        self.assertIn("MainlineAgent", result.route_plan_text)
        self.assertIn("NoveltyAgent", result.route_plan_text)
        self.assertTrue(result.focus_points)


if __name__ == "__main__":
    unittest.main()