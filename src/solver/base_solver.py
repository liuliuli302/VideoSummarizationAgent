"""Common base class for baseline training/inference/evaluation tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch

from src.config import build_runtime_config


class BaseSolver(ABC):
    """Shared runtime utilities for CLI tasks."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = build_runtime_config(config)
        self.device = self._resolve_device(str(self.config.experiment.device))
        self._set_seed(int(self.config.experiment.seed or self.config.project.seed))
        self._setup_logging()

    def _resolve_device(self, device_name: str) -> torch.device:
        if device_name == "cpu":
            return torch.device("cpu")
        if device_name.startswith("cuda") and torch.cuda.is_available():
            return torch.device(device_name)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _setup_logging(self) -> None:
        print(f"{self.__class__.__name__} initialized on {self.device}")

    @abstractmethod
    def run(self):
        """Execute the solver task."""
