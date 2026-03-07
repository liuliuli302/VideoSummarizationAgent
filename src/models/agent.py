"""High-level baseline video agent wrapper."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from src.models.networks import AgentCore, VisionEncoder
from src.utils.tools import VideoTools


class VideoAgent:
    """Simple perception-policy agent used by legacy baseline experiments."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        model_config = self.config.get("model_config", {})
        agent_config = self.config.get("agent_config", {})

        self.vision_encoder = VisionEncoder(model_config)
        visual_dim = getattr(self.vision_encoder, "embed_dim", int(model_config.get("embed_dim", 512)))
        self.core_policy = AgentCore(visual_dim=visual_dim, **agent_config)
        self.tools = VideoTools()
        self.current_state_feat: torch.Tensor | None = None

    def perceive(self, observation: Dict[str, torch.Tensor]) -> None:
        """Encode the current video observation into latent state features."""
        video_tensor = observation.get("video")
        if video_tensor is None:
            raise ValueError("observation must contain a `video` tensor")

        device = next(self.vision_encoder.parameters()).device
        if video_tensor.device != device:
            video_tensor = video_tensor.to(device)
        if video_tensor.ndim == 4:
            video_tensor = video_tensor.unsqueeze(0)

        self.current_state_feat = self.vision_encoder(video_tensor)

    def act(self) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        """Predict frame-level actions and probabilities."""
        if self.current_state_feat is None:
            return None, None

        logits = self.core_policy(self.current_state_feat)
        probabilities = F.softmax(logits, dim=-1)
        actions = torch.argmax(probabilities, dim=-1)
        return actions, probabilities
