import unittest
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.agent import VideoAgent

class TestVideoAgent(unittest.TestCase):
    def setUp(self):
        # Config for ResNet18 (default)
        self.config = {
            'model_config': {'embed_dim': 512, 'model_name': 'resnet18'},
            'agent_config': {'hidden_dim': 256, 'action_space': 2},
            'video': {'num_frames': 16}
        }
        self.agent = VideoAgent(self.config)

    def test_agent_initialization(self):
        self.assertIsNotNone(self.agent.vision_encoder)
        self.assertIsNotNone(self.agent.core_policy)
        # Check inferred dimension
        input_dim = self.agent.core_policy.temporal_encoder.input_size
        self.assertEqual(input_dim, 512, "Visual dim should match ResNet output (512)")

    def test_perceive_and_act(self):
        # Inputs: [Time=16, C=3, H=224, W=224]
        # Agent will add batch dimension
        video_tensor = torch.randn(16, 3, 224, 224)
        observation = {"video": video_tensor}
        
        self.agent.perceive(observation)
        action, probs = self.agent.act()
        
        # Expect batch size 1 since input was single video
        # Output: [Batch=1, Time=16]
        self.assertEqual(action.dim(), 2)
        self.assertEqual(action.shape[0], 1)
        self.assertEqual(action.shape[1], 16)
        
        # Probs: [Batch, Time, ActionSpace=2]
        self.assertEqual(probs.shape[-1], 2)
        
        print(f"Action shape: {action.shape}, Probs shape: {probs.shape}")

if __name__ == '__main__':
    unittest.main()
