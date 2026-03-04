import unittest
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.agent import VideoAgent
from src.utils.video_utils import load_video_frames

class TestVideoAgent(unittest.TestCase):
    def setUp(self):
        self.config = {
            'model_config': {'embed_dim': 768},
            'agent_config': {'hidden_dim': 256}
        }
        self.agent = VideoAgent(self.config)

    def test_agent_initialization(self):
        self.assertIsNotNone(self.agent.vision_encoder)
        self.assertIsNotNone(self.agent.core_policy)

    def test_perceive_and_act(self):
        # 创建随机视频张量 [Time, Channel, H, W]
        video_tensor = torch.randn(16, 3, 224, 224)
        observation = {"video": video_tensor}
        
        self.agent.perceive(observation)
        action, probs = self.agent.act()
        
        self.assertIsInstance(action, int)
        self.assertEqual(probs.shape[-1], 5) # 默认 action_space=5

if __name__ == '__main__':
    unittest.main()
