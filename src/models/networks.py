"""Neural baseline modules for frame-level video importance prediction."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torchvision.models as models


class VisionEncoder(nn.Module):
    """Encode sampled video frames into per-frame embeddings.

    Input shape:
        `[B, T, C, H, W]`
    Output shape:
        `[B, T, D]`
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__()
        config = config or {}
        self.embed_dim = int(config.get("embed_dim", 512))
        self.model_name = str(config.get("model_name", "resnet18"))
        use_pretrained = bool(config.get("pretrained", False))

        if self.model_name != "resnet18":
            raise ValueError(f"Unsupported model_name: {self.model_name}")

        weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
        backbone = models.resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        backbone_dim = backbone.fc.in_features
        self.projector = nn.Identity() if backbone_dim == self.embed_dim else nn.Linear(backbone_dim, self.embed_dim)

    def forward(self, video_tensor: torch.Tensor) -> torch.Tensor:
        if video_tensor.ndim != 5:
            raise ValueError(
                f"video_tensor must have shape [B, T, C, H, W], got {tuple(video_tensor.shape)}"
            )

        batch_size, num_frames, channels, height, width = video_tensor.shape
        flattened = video_tensor.reshape(batch_size * num_frames, channels, height, width)
        features = self.feature_extractor(flattened).flatten(1)
        projected = self.projector(features)
        return projected.reshape(batch_size, num_frames, -1)


class AgentCore(nn.Module):
    """Temporal policy head that predicts frame-level importance logits."""

    def __init__(
        self,
        visual_dim: int = 512,
        hidden_dim: int = 256,
        action_space: int = 2,
        num_layers: int = 2,
    ):
        super().__init__()
        if action_space <= 0:
            raise ValueError(f"action_space must be positive, got {action_space}")

        self.temporal_encoder = nn.LSTM(
            input_size=visual_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space),
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        if visual_features.ndim != 3:
            raise ValueError(
                f"visual_features must have shape [B, T, D], got {tuple(visual_features.shape)}"
            )
        encoded_features, _ = self.temporal_encoder(visual_features)
        return self.head(encoded_features)
