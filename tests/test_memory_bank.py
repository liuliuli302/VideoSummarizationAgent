import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.schemas import WindowFeature
from src.memory.memory_bank import MemoryBank
from src.memory.retrieval import retrieve_relevant_texts


class TestMemoryBankStep7(unittest.TestCase):
    def setUp(self):
        self.memory_bank = MemoryBank(topk=2, max_items_per_slot=3)
        self.feature_a = WindowFeature(
            win_id="w_0",
            visual_description="A city street with moving traffic.",
            local_caption="A busy city street scene.",
            semantic_summary="Local event: A busy city street scene with moving traffic.",
            evidence_notes=["street", "traffic"],
        )
        self.feature_b = WindowFeature(
            win_id="w_1",
            visual_description="A classroom lecture with slides.",
            local_caption="A lesson is being explained.",
            semantic_summary="Local event: A lesson is being explained with slides.",
            evidence_notes=["lesson", "slides"],
        )
        self.feature_c = WindowFeature(
            win_id="w_2",
            visual_description="A city crowd walking downtown.",
            local_caption="Pedestrians move through a crowded street.",
            semantic_summary="Local event: Pedestrians move through a crowded downtown street.",
            evidence_notes=["crowd", "street"],
        )

    def test_retrieve_relevant_texts(self):
        texts = [
            "busy city street traffic",
            "classroom lecture slides",
            "crowded downtown street",
        ]
        result = retrieve_relevant_texts(texts, query="downtown street", topk=2)
        self.assertEqual(result[0], "crowded downtown street")
        self.assertEqual(len(result), 2)

    def test_update_and_read_memory(self):
        self.memory_bank.update(self.feature_a, final_decision="建议保留", is_selected=True)
        self.memory_bank.update(self.feature_b, final_decision="可选", is_selected=False)
        self.memory_bank.update(self.feature_c, final_decision="建议保留", is_selected=True)

        read_result = self.memory_bank.read("downtown street traffic")

        self.assertTrue(read_result["selected_ctx"])
        self.assertIn(self.feature_c.semantic_summary, read_result["selected_ctx"][0])
        self.assertEqual(len(read_result["temporal_ctx"]), 2)

    def test_retrieve_topk_specific_slot(self):
        self.memory_bank.update(self.feature_a, final_decision="建议保留", is_selected=True)
        self.memory_bank.update(self.feature_b, final_decision="建议保留", is_selected=True)

        result = self.memory_bank.retrieve_topk("story_slots", query="lecture slides", topk=1)
        self.assertEqual(result, [self.feature_b.semantic_summary])

    def test_capacity_trim(self):
        extra_feature = WindowFeature(
            win_id="w_3",
            visual_description="A sports match.",
            local_caption="Players run on a field.",
            semantic_summary="Local event: Players run on a field during a sports match.",
            evidence_notes=["sports"],
        )

        self.memory_bank.update(self.feature_a, final_decision="建议保留", is_selected=True)
        self.memory_bank.update(self.feature_b, final_decision="建议保留", is_selected=True)
        self.memory_bank.update(self.feature_c, final_decision="建议保留", is_selected=True)
        self.memory_bank.update(extra_feature, final_decision="建议保留", is_selected=True)

        snapshot = self.memory_bank.snapshot()
        self.assertEqual(len(snapshot.story_slots), 3)
        self.assertNotIn(self.feature_a.semantic_summary, snapshot.story_slots)


if __name__ == "__main__":
    unittest.main()