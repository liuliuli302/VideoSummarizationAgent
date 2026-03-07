"""Single-video inference solver for the legacy `VideoAgent` baseline."""

from __future__ import annotations

from typing import Any, Dict

from src.models.agent import VideoAgent
from src.solver.base_solver import BaseSolver
from src.utils.video_utils import load_video_frames


class InferenceSolver(BaseSolver):
    """Run baseline frame-level inference on a single video."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.agent = VideoAgent(self.config)
        self.agent.vision_encoder.to(self.device)
        self.agent.core_policy.to(self.device)
        self.video_path = self.config.get("video_path", "data/raw/demo.mp4")
        self.num_frames = int(self.config.get("video", {}).get("num_frames", 16))

    def run(self) -> None:
        print(f"Running inference on {self.video_path}")
        video_tensor = load_video_frames(self.video_path, num_frames=self.num_frames).to(self.device)
        self.agent.perceive({"video": video_tensor})
        action, probabilities = self.agent.act()
        print(f"Predicted Action: {action}")
        print(f"Probability: {probabilities}")
