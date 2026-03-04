import torch
from torch.utils.data import DataLoader
from src.solver.base_solver import BaseSolver
from src.data.dataset import VideoDataset
from src.models.agent import VideoAgent



class ExperimentSolver(BaseSolver):
    def __init__(self, config):
        super().__init__(config)
        self.dataset = VideoDataset(data_root=config.get('data_root', "data/raw/"))
        self.dataloader = DataLoader(self.dataset, batch_size=config.get('batch_size', 2), shuffle=True)
        
        agent_config = config.get('agent_config', {})
        self.agent = VideoAgent(agent_config)

    def run(self):
        iterations = self.config.get('iterations', 1)
        print("Starting experiment loop (Training-Free)...")
        
        # Ensure agent is on device
        if hasattr(self, 'device'):
            if hasattr(self.agent, 'vision_encoder'): self.agent.vision_encoder.to(self.device)
            if hasattr(self.agent, 'core_policy'): self.agent.core_policy.to(self.device)

        for i_iter in range(iterations): 
            for i, batch in enumerate(self.dataloader):
                videos = batch['video']
                if hasattr(self, 'device'):
                     videos = videos.to(self.device)
                
                # --- Forward / Process ---
                observation = {"video": videos}
                self.agent.perceive(observation)
                action, probs = self.agent.act()
                
                print(f"Iteration {i_iter}, Step {i}, Action: {action}")

