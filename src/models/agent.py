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
        
        # 初始化视觉模块
        vision_cfg = config.get('model_config', {})
        self.vision_encoder = VisionEncoder(vision_cfg)
        
        # 初始化核心决策模块
        agent_cfg = config.get('agent_config', {})
        visual_dim = vision_cfg.get('embed_dim', 768)
        self.core_policy = AgentCore(visual_dim=visual_dim)
        
        # 工具集
        self.tools = VideoTools()
        
    def perceive(self, observation):
        """
        observation: dict
          - video_tensor: [T, C, H, W]
        """
        video_tensor = observation.get('video')
        # 增加batch维度并传入
        # Note: In real scenarios, ensure device placement
        if hasattr(self.vision_encoder, 'parameters') and next(self.vision_encoder.parameters()).is_cuda:
            video_tensor = video_tensor.to(next(self.vision_encoder.parameters()).device)

        
        # Determine if we have a batch dimension
        if video_tensor.dim() == 4:
            video_tensor = video_tensor.unsqueeze(0)

        video_feat = self.vision_encoder(video_tensor) 
        
        # 实际上可能还要融合文本指令 (Text/Instruction)
        # 这里简化为只用视觉特征
        self.current_state_feat = video_feat
        
    def act(self):
        """
        返回动作分布或具体动作
        """
        if not hasattr(self, 'current_state_feat'):
            return None
        
        logits = self.core_policy(self.current_state_feat)
        probs = F.softmax(logits, dim=-1)
        action = torch.argmax(probs, dim=-1)
        
        # 模拟调用工具 (只对单样本或者第一个样本演示)
        if action.dim() == 0:
             # Single sample
             if action.item() == 0:
                print("Agent decided to extract a clip using tools.")
                # dummy clip
                _ = self.tools.extract_clip(torch.randn(16, 3, 224, 224), 0, 5)
             return action.item(), probs
        else:
             # Batch - return tensor
             return action, probs

