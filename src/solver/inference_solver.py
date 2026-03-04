import torch
from src.solver.base_solver import BaseSolver
from src.models.agent import VideoAgent


from src.utils.video_utils import load_video_frames

class InferenceSolver(BaseSolver):
    def __init__(self, config):
        super().__init__(config)
        agent_config = config.get('agent_config', {})
        self.agent = VideoAgent(agent_config)
        self.video_path = config.get('video_path', "data/raw/demo.mp4")

    def run(self):
        print(f"Running inference on {self.video_path}")
        self.agent.vision_encoder.to(self.device)
        self.agent.core_policy.to(self.device)
        
        # Load Video
        video_tensor = load_video_frames(self.video_path).to(self.device)
        observation = {"video": video_tensor}
        
        # Perceive
        # Note: In real perception, we might need to handle batch dim if the agent expects it
        # The original code did manual perceive then act. 
        # But VideoAgent.perceive expects dictionary.
        
        # Because we moved to device, ensure agent handles it or we pass it correctly
        # The original agent.perceive:
        # video_tensor = observation.get('video')
        # video_feat = self.vision_encoder(video_tensor.unsqueeze(0)) 
        
        # We need to make sure the tensor is on the correct device, which we did above.
        
        self.agent.perceive(observation)
        
        action, probs = self.agent.act()
        print(f"Predicted Action: {action}")
        print(f"Probability: {probs}")
