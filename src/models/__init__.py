"""Baseline model package used by legacy experiment/inference/eval tasks."""

from src.models.agent import VideoAgent
from src.models.networks import AgentCore, VisionEncoder

__all__ = ["VisionEncoder", "AgentCore", "VideoAgent"]
