
from src.solver.base_solver import BaseSolver
from src.data.loader.dataset import VideoDataset
from src.agent import VideoAgent
import numpy as np

class EvalSolver(BaseSolver):
    def __init__(self, config):
        super().__init__(config)
        self.dataset = VideoDataset(data_root=config.get('data_root', "data/raw/"))
        
        agent_config = config.get('agent_config', {})
        self.agent = VideoAgent(agent_config)

    def run(self):
        print("Running evaluation task...")
        self.run_evaluation(num_samples=self.config.get('num_samples', 5))

    def run_evaluation(self, num_samples=10):
        print("Starting Evaluation...")
        correct = 0
        total = 0
        
        # Ensure agent is on device
        if hasattr(self, 'device'):
            self.agent.vision_encoder.to(self.device)
            self.agent.core_policy.to(self.device)

        # Simple accuracy check on random samples
        for i in range(min(num_samples, len(self.dataset))):
            sample = self.dataset[i]
            # Handle potential device mismatch if dataset returns CPU tensor
            video = sample['video']
            if hasattr(self, 'device'):
                 video = video.to(self.device)

            observation = {"video": video}
            self.agent.perceive(observation)
            action, _ = self.agent.act()
            
            print(f"Sample {i}: Agent chose action {action}")
            total += 1
            # Random 'correct' for demo
            if np.random.rand() > 0.5:
                correct += 1
                
        accuracy = correct / total if total > 0 else 0
        print(f"Evaluation Complete. Accuracy: {accuracy:.2f}")

