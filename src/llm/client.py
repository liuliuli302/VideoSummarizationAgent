from __future__ import annotations

import json
import os
import re
import time
from base64 import b64encode
from io import BytesIO
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image


load_dotenv()


class LLMClient:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        if not resolved_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
        self.max_retries = 3

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        response = self._chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""

    def generate_multimodal_text(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[Image.Image],
    ) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._to_data_url(image),
                    },
                }
            )

        response = self._chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )
        content_text = response.choices[0].message.content
        return content_text.strip() if content_text else ""

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raw_text = self.generate_text(system_prompt=system_prompt, user_prompt=user_prompt)
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            retry_prompt = (
                f"{user_prompt}\n\n"
                "Your previous response was not valid JSON. Return valid JSON only."
            )
            retry_text = self.generate_text(system_prompt=system_prompt, user_prompt=retry_prompt)
            try:
                return json.loads(retry_text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"LLM output is not valid JSON: {retry_text}") from exc

    def _to_data_url(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        encoded = b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def _chat_completion(self, messages: list[dict[str, Any]]):
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=messages,
                )
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    raise
                time.sleep(1.5 * attempt)
        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM request failed without a captured exception.")


class MockLLMClient:
    def __init__(self, model: str = "mock", temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        return json.dumps(self.generate_json(system_prompt=system_prompt, user_prompt=user_prompt), ensure_ascii=False)

    def generate_multimodal_text(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[Image.Image],
    ) -> str:
        return (
            f"A video segment with {len(images)} sampled frames showing continuous visual activity, scene context, and temporal progression."
        )

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        prompt = user_prompt.lower()
        if "infer the video theme" in prompt:
            theme = self._infer_theme(prompt)
            return {
                "video_theme": theme,
                "global_summary": self._mock_global_summary(user_prompt),
                "expert_weights": self._mock_weights(theme),
                "reason": f"Mock planner inferred the theme as {theme} from the segment captions.",
            }

        if "evaluate the importance of the current segment" in prompt:
            score = self._score_from_text(prompt, bias=0.45)
            return {
                "score": score,
                "reason": "Mock planner score based on the current caption content and lightweight keyword heuristics.",
            }

        score = self._score_from_text(prompt, bias=0.35)
        return {
            "score": score,
            "reason": "Mock expert score based on a deterministic keyword heuristic for local debugging.",
        }

    def _infer_theme(self, text: str) -> str:
        if any(word in text for word in ["goal", "match", "race", "team", "sport"]):
            return "sports"
        if any(word in text for word in ["tutorial", "lesson", "teach", "explain"]):
            return "tutorial"
        if any(word in text for word in ["travel", "street", "city", "vlog"]):
            return "vlog"
        if any(word in text for word in ["landscape", "mountain", "sea", "nature"]):
            return "landscape"
        if any(word in text for word in ["story", "dialogue", "character", "plot"]):
            return "story"
        return "general"

    def _mock_global_summary(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip().startswith("Segment")]
        summary_source = " ".join(lines[:3])
        cleaned = re.sub(r"\s+", " ", summary_source).strip()
        return cleaned[:300] if cleaned else "Mock global summary from segment captions."

    def _mock_weights(self, theme: str) -> dict[str, float]:
        presets = {
            "story": {"story_agent": 0.45, "visual_agent": 0.15, "emotion_agent": 0.25, "information_agent": 0.15},
            "tutorial": {"story_agent": 0.1, "visual_agent": 0.15, "emotion_agent": 0.1, "information_agent": 0.65},
            "landscape": {"story_agent": 0.1, "visual_agent": 0.6, "emotion_agent": 0.2, "information_agent": 0.1},
            "sports": {"story_agent": 0.25, "visual_agent": 0.35, "emotion_agent": 0.25, "information_agent": 0.15},
            "vlog": {"story_agent": 0.25, "visual_agent": 0.25, "emotion_agent": 0.2, "information_agent": 0.3},
            "general": {"story_agent": 0.25, "visual_agent": 0.25, "emotion_agent": 0.25, "information_agent": 0.25},
        }
        return presets.get(theme, presets["general"])

    def _score_from_text(self, text: str, bias: float) -> float:
        bonus = 0.0
        keywords = [
            "important", "highlight", "goal", "change", "people", "action", "emotion",
            "instruction", "explain", "scene", "movement", "event", "object",
        ]
        for keyword in keywords:
            if keyword in text:
                bonus += 0.04
        bonus += min(text.count("segment"), 3) * 0.03
        return max(0.0, min(1.0, round(bias + bonus, 4)))


def build_llm_client(
    llm_mode: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.2,
) -> LLMClient | MockLLMClient:
    if llm_mode == "mock":
        return MockLLMClient(model=model, temperature=temperature)
    return LLMClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )