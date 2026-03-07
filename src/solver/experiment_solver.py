"""Baseline experiment loop for quick training or dry-run inference."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataload.dataset import VideoDataset
from src.models.agent import VideoAgent
from src.solver.base_solver import BaseSolver


class ExperimentSolver(BaseSolver):
    """Train or dry-run the legacy `VideoAgent` baseline."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        video_config = self.config.get("video", {})
        training_config = self.config.get("training", {})

        self.dataset = VideoDataset(
            data_root=self.config.get("data_root", "data/raw"),
            metadata_file=self.config.get("metadata_file"),
            num_frames=int(video_config.get("num_frames", 16)),
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=int(training_config.get("batch_size", 2)),
            shuffle=True,
        )
        self.agent = VideoAgent(self.config)
        self.agent.vision_encoder.to(self.device)
        self.agent.core_policy.to(self.device)

        self.mode = str(self.config.get("mode", "inference"))
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            self.agent.core_policy.parameters(),
            lr=float(training_config.get("lr", 1e-4)),
        )

    def run(self) -> None:
        epochs = int(self.config.get("training", {}).get("epochs", self.config.get("epochs", 1)))
        print(f"Starting {self.mode} loop for {epochs} epoch(s)")

        for epoch_idx in range(epochs):
            epoch_loss = 0.0
            num_steps = 0
            for step_idx, batch in enumerate(self.dataloader):
                video_batch = batch["video"].to(self.device)
                label_batch = batch["label"].to(self.device).float()
                self.agent.perceive({"video": video_batch})

                if self.mode == "train":
                    logits = self.agent.core_policy(self.agent.current_state_feat)
                    loss = self._compute_loss(logits, label_batch)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += float(loss.item())
                    num_steps += 1

                    if step_idx % 10 == 0:
                        print(f"Epoch {epoch_idx} Step {step_idx} Loss {loss.item():.4f}")
                else:
                    action, probabilities = self.agent.act()
                    print(f"Epoch {epoch_idx} Step {step_idx} Action shape: {tuple(action.shape)}")
                    print(f"Epoch {epoch_idx} Step {step_idx} Probability shape: {tuple(probabilities.shape)}")

            if self.mode == "train" and num_steps > 0:
                print(f"Epoch {epoch_idx} finished. Avg Loss: {epoch_loss / num_steps:.4f}")

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] == 1:
            return self.criterion(logits.squeeze(-1), labels)
        labels = labels.long().clamp_min(0)
        if labels.ndim == 2:
            return nn.CrossEntropyLoss()(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return nn.CrossEntropyLoss()(logits, labels)
