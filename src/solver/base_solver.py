import os
import yaml
import torch
from abc import ABC, abstractmethod

class BaseSolver(ABC):
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_logging()

    def _setup_logging(self):
        print(f"Solver initialized on {self.device}")

    @abstractmethod
    def run(self):
        pass
