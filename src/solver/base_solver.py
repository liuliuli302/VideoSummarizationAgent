"""Common base class for baseline training/inference/evaluation tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class BaseSolver(ABC):
    """Shared runtime utilities for CLI tasks."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_logging()

    def _setup_logging(self) -> None:
        print(f"{self.__class__.__name__} initialized on {self.device}")

    @abstractmethod
    def run(self):
        """Execute the solver task."""
