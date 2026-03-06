from __future__ import annotations

from typing import Optional

from src.datasets.schemas import Window
from src.perception.visual_encoder import WindowVisualEncoder


class RuleBasedCaptioner:
    """Generate a stable local caption from sampled frames."""

    def __init__(self, visual_encoder: Optional[WindowVisualEncoder] = None):
        self.visual_encoder = visual_encoder or WindowVisualEncoder()

    def generate_caption(self, window: Window, video_path: Optional[str] = None, frames=None) -> str:
        analysis = self.visual_encoder.analyze_window(window=window, video_path=video_path, frames=frames)

        brightness_phrase = self._brightness_phrase(analysis["brightness"])
        color_phrase = self._color_phrase(analysis["dominant_color"])
        motion_phrase = self._motion_phrase(analysis["motion"])
        variation_phrase = self._variation_phrase(analysis["scene_variation"])

        return (
            f"A {brightness_phrase} {color_phrase} scene with {motion_phrase} motion and "
            f"{variation_phrase} visual change."
        )

    def _brightness_phrase(self, brightness: str) -> str:
        mapping = {
            "dark": "low-light",
            "balanced": "well-lit",
            "bright": "bright",
        }
        return mapping.get(brightness, brightness)

    def _color_phrase(self, dominant_color: str) -> str:
        mapping = {
            "red": "red-dominant",
            "green": "green-dominant",
            "blue": "blue-dominant",
            "balanced": "color-balanced",
        }
        return mapping.get(dominant_color, dominant_color)

    def _motion_phrase(self, motion: str) -> str:
        mapping = {
            "static": "minimal",
            "moderate": "moderate",
            "dynamic": "strong",
        }
        return mapping.get(motion, motion)

    def _variation_phrase(self, variation: str) -> str:
        mapping = {
            "stable": "limited",
            "changing": "noticeable",
            "highly changing": "rapid",
        }
        return mapping.get(variation, variation)