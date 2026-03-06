import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents import DomainAgent, EventAgent, MainlineAgent, NoveltyAgent, TemporalAgent
from src.datasets.schemas import MemoryState, WindowFeature
from src.scoring.aggregator import AggregationAgent


class TestRuleAgentsStep8(unittest.TestCase):
    def setUp(self):
        self.window_feature = WindowFeature(
            win_id="w_5",
            visual_description="Window w_5: sampled 4 frames from 20 to 28; brightness=bright; dominant_color=red; motion=dynamic; scene_variation=changing.",
            local_caption="A bright red-dominant scene with strong motion and noticeable visual change.",
            semantic_summary="Visual Description: bright travel street activity.\nLocal Caption: pedestrians move through a busy downtown street.\nSemantic Summary: Local event: pedestrians move through a busy downtown street.",
            evidence_notes=["street", "crowd"],
        )
        self.memory_state = MemoryState(
            story_slots=["Local event: arrival in the city center."],
            selected_slots=["Local event: quiet indoor explanation with slides."],
            redundancy_bank=[],
            temporal_context=["Local event: entering a crowded street.", "Local event: walking past downtown shops."],
        )
        self.summary_goal = (
            "Primary theme: travel. Summary goal: preserve the main progression across coarse segments "
            "and retain representative city activities."
        )

    def test_mainline_agent_output(self):
        agent = MainlineAgent()
        output = agent.run(
            window_summary=self.window_feature.semantic_summary,
            summary_goal=self.summary_goal,
            story_memory=self.memory_state.story_slots,
        )
        self.assertIn("Mainline Judgment:", output)
        self.assertIn("Conclusion:", output)
        self.assertIn("Reason:", output)

    def test_other_agents_and_aggregator(self):
        novelty = NoveltyAgent().run(
            window_summary=self.window_feature.semantic_summary,
            selected_memory=self.memory_state.selected_slots,
            recent_context=self.memory_state.temporal_context,
        )
        event = EventAgent().run(
            visual_description=self.window_feature.visual_description,
            local_caption=self.window_feature.local_caption,
            asr_text="The street gets busier as we move downtown.",
        )
        temporal = TemporalAgent().run(
            prev_summary=self.memory_state.temporal_context[0],
            current_summary=self.window_feature.semantic_summary,
            next_summary="Local event: the camera turns toward a busy plaza.",
            summary_chain=self.memory_state.temporal_context,
        )
        domain = DomainAgent().run(
            video_theme="travel vlog",
            domain_hint="city exploration and moving between landmarks",
            window_summary=self.window_feature.semantic_summary,
        )
        mainline = MainlineAgent().run(
            window_summary=self.window_feature.semantic_summary,
            summary_goal=self.summary_goal,
            story_memory=self.memory_state.story_slots,
        )

        aggregator = AggregationAgent()
        result = aggregator.run(
            win_id=self.window_feature.win_id,
            expert_outputs={
                "mainline": mainline,
                "novelty": novelty,
                "event": event,
                "temporal": temporal,
                "domain": domain,
            },
        )

        self.assertIn("Novelty:", novelty)
        self.assertIn("Event Importance:", event)
        self.assertIn("Temporal Value:", temporal)
        self.assertIn("Domain Match:", domain)
        self.assertEqual(result.win_id, "w_5")
        self.assertIn("mainline", result.expert_opinions)
        self.assertIn(result.final_importance, {"必须保留", "建议保留", "可选", "建议省略"})
        self.assertIn("Final Importance:", result.base_decision)


if __name__ == "__main__":
    unittest.main()