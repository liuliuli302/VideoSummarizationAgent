"""Evaluation solver for the legacy `VideoAgent` baseline."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import torch

from src.dataload.dataset import VideoDataset
from src.models.agent import VideoAgent
from src.solver.base_solver import BaseSolver


class EvalSolver(BaseSolver):
    """Evaluate the baseline agent against frame-level labels when available."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        video_config = self.config.video
        self.dataset = VideoDataset(
            data_root=self.config.dataset.data_root,
            metadata_file=self.config.dataset.metadata_file,
            num_frames=int(video_config.num_frames),
        )
        self.agent = VideoAgent(self.config)
        self.agent.vision_encoder.to(self.device)
        self.agent.core_policy.to(self.device)
        self.output_dir = self.config.evaluation.output_dir

    def run(self) -> Dict[str, float]:
        num_samples = int(self.config.evaluation.num_samples)
        metrics = self.run_evaluation(num_samples=num_samples)
        self._save_metrics(metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return metrics

    def run_evaluation(self, num_samples: int = 10) -> Dict[str, float]:
        sample_count = min(num_samples, len(self.dataset))
        if sample_count == 0:
            return {"samples": 0, "frame_accuracy": 0.0, "positive_rate": 0.0}

        total_frames = 0
        correct_frames = 0
        positive_predictions = 0

        with torch.no_grad():
            for sample_idx in range(sample_count):
                sample = self.dataset[sample_idx]
                video_tensor = sample["video"].to(self.device)
                label_tensor = sample["label"].to(self.device)
                self.agent.perceive({"video": video_tensor})
                actions, _ = self.agent.act()
                predicted = actions.squeeze(0).float()
                target = (label_tensor > 0).float()

                if predicted.numel() != target.numel():
                    min_length = min(predicted.numel(), target.numel())
                    predicted = predicted[:min_length]
                    target = target[:min_length]

                correct_frames += int((predicted == target).sum().item())
                total_frames += int(target.numel())
                positive_predictions += int(predicted.sum().item())

        return {
            "samples": sample_count,
            "frame_accuracy": round(correct_frames / max(1, total_frames), 4),
            "positive_rate": round(positive_predictions / max(1, total_frames), 4),
        }

    def _save_metrics(self, metrics: Dict[str, float]) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, "baseline_eval.json")
        with open(output_path, "w", encoding="utf-8") as file_obj:
            json.dump(metrics, file_obj, ensure_ascii=False, indent=2)
