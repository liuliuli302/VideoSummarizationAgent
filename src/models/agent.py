import torch
import torch.nn.functional as F
from src.utils.tools import VideoTools
from src.models.networks import VisionEncoder, AgentCore

class VideoAgent:
    def __init__(self, config):
        """
        config: dict
          - model_config: 视觉模型配置
          - agent_config: 策略配置
        """
        self.config = config
        
        # Initialize Vision Encoder
        model_cfg = config.get('model_config', {})
        self.vision_encoder = VisionEncoder(model_cfg)
        
        # Initialize Core Policy
        agent_cfg = config.get('agent_config', {})
        
        # Dynamic dimension check
        if hasattr(self.vision_encoder, 'embed_dim'):
            visual_dim = self.vision_encoder.embed_dim
        else:
            visual_dim = model_cfg.get('embed_dim', 768)
        
        # Pass remaining agent config (action_space, hidden_dim, etc.)
        self.core_policy = AgentCore(visual_dim=visual_dim, **agent_cfg)
        
        # Tools
        self.tools = VideoTools()
        
    def perceive(self, observation):
        """
        observation: dict
          - video: [Batch, Time, C, H, W] or [Time, C, H, W]
        """
        video_tensor = observation.get('video')
        if video_tensor is None:
            return
            
        # Ensure device placement
        device = next(self.vision_encoder.parameters()).device
        if video_tensor.device != device:
            video_tensor = video_tensor.to(device)

        # Ensure batch dimension
        if video_tensor.dim() == 4:
            video_tensor = video_tensor.unsqueeze(0)

        # [Batch, Time, EmbedDim]
        video_feat = self.vision_encoder(video_tensor) 
        
        # Store state
        self.current_state_feat = video_feat
        
    def act(self):
        """
        Returns action distribution and selected action.
        Output:
            action: [Batch, Time] (indices)
            probs: [Batch, Time, ActionSpace]
        """
        if not hasattr(self, 'current_state_feat'):
            return None, None
        
        # [Batch, Time, ActionSpace]
        logits = self.core_policy(self.current_state_feat)
        probs = F.softmax(logits, dim=-1)
        action = torch.argmax(probs, dim=-1)
        
        return action, probs

