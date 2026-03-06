import torch
from torch.utils.data import DataLoader
from src.solver.base_solver import BaseSolver
from src.dataload.dataset import VideoDataset
from src.models.agent import VideoAgent



import torch.optim as optim
import torch.nn as nn

class ExperimentSolver(BaseSolver):
    def __init__(self, config):
        """
        Experiment Solver mainly for inference or dry-run loops.
        """
        super().__init__(config)
        
        # Load Video Config
        video_conf = config.get('video', {})
        num_frames = video_conf.get('num_frames', 16)
        data_root = config.get('data_root', "data/raw/")
        
        print(f"Initializing Dataset with root: {data_root}, num_frames: {num_frames}")
        self.dataset = VideoDataset(
            data_root=data_root, 
            num_frames=num_frames
        )
        
        train_cfg = config.get('training', {})
        batch_size = train_cfg.get('batch_size', 2)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        # Model
        self.agent = VideoAgent(config)
        
        # Training Setup
        self.mode = config.get('mode', 'inference')
        if self.mode == 'train':
            lr = train_cfg.get('lr', 1e-4)
            # Optimize only policy for now, keep vision frozen or slow?
            # For simplicity, optimize policy head only as ResNet is fixed/pretrained usually
            # unless fine-tuning.
            params = list(self.agent.core_policy.parameters())
            # If we want to fine-tune vision:
            # params += list(self.agent.vision_encoder.parameters())
            
            self.optimizer = optim.Adam(params, lr=lr)
            # Binary Classification for importance (0 or 1)
            # Agent output raw logits (B, T, 1) or (B, T, 2)
            # If default logic: AgentCore outputs (B, T, 1)
            self.criterion = nn.BCEWithLogitsLoss()

    def run(self):
        iterations = self.config.get('iterations', 1) # or epochs
        epochs = self.config.get('training', {}).get('epochs', 1)
        
        status_msg = "Training" if self.mode == 'train' else "Inference"
        print(f"Starting {status_msg} loop...")
        
        # Ensure agent is on device
        device = self.device if hasattr(self, 'device') else torch.device('cpu')
        if hasattr(self.agent, 'vision_encoder'): self.agent.vision_encoder.to(device)
        if hasattr(self.agent, 'core_policy'): self.agent.core_policy.to(device)

        for epoch in range(epochs):
            total_loss = 0
            for i, batch in enumerate(self.dataloader):
                videos = batch['video'].to(device) # [B, T, C, H, W]
                labels = batch['label'].to(device) # [B, T]
                
                # --- Forward / Process ---
                observation = {"video": videos}
                self.agent.perceive(observation)
                
                # Action: [B, T], Probs: [B, T, A]
                # We need raw logits for training usually
                # But act() applies softmax.
                # Let's access logits directly from core_policy for training
                if self.mode == 'train':
                    logits = self.agent.core_policy(self.agent.current_state_feat) # [B, T, 1] or [B, T, A]
                    
                    # Assume binary classification
                    # Using BCEWithLogitsLoss, target needs to be float matching input
                    if logits.shape[-1] == 1:
                        loss = self.criterion(logits.squeeze(-1), labels.float())
                    else:
                        # CrossEntropy
                        # labels long
                        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.shape[-1]), labels.view(-1).long())
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    if i % 10 == 0:
                        print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")

                else:
                    # Inference
                    action, probs = self.agent.act()
                    print(f"Step {i}, Action: {action[0] if action is not None else 'None'}")
            
            if self.mode == 'train':
                 print(f"Epoch {epoch} finished. Avg Loss: {total_loss / len(self.dataloader):.4f}")


