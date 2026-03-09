"""Real DeepSeek API client used for scene and video caption generation."""

from __future__ import annotations

import os
from typing import Iterable, List

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

_API_KEY = os.getenv("OPENAI_API_KEY")
client = (
    OpenAI(
        api_key=_API_KEY,
        base_url="https://www.dmxapi.cn/v1",
    )
    if _API_KEY
    else None
)


def call_llm(prompt: str, temperature: float = 0.3) -> str:
    if client is None:
        raise RuntimeError("OPENAI_API_KEY not found; real API-backed captioning is unavailable.")
    response = client.chat.completions.create(
        model="DeepSeek-V3.2",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    if isinstance(response, str):
        preview = response[:200].replace("\n", " ")
        raise RuntimeError(f"Unexpected non-JSON response from LLM endpoint: {preview}")
    content = response.choices[0].message.content
    return content.strip() if content else ""


class DeepSeekVideoCaptioner:
    """Real API-backed caption generator for scenes and video-level aggregation."""

    def __init__(self, temperature: float = 0.3) -> None:
        if client is None:
            raise RuntimeError("OPENAI_API_KEY not found; DeepSeekVideoCaptioner cannot be initialized.")
        self.temperature = temperature

    def caption_scene(
        self,
        video_title: str | None,
        category: str | None,
        scene_index: int,
        start_frame: int,
        end_frame: int,
        keyframe_descriptions: Iterable[str],
    ) -> str:
        print("Calling LLM for scene caption...")
        keyframes_text = "\n".join(f"- {item}" for item in keyframe_descriptions if item)
        prompt = (
            "You are a video scene captioning assistant. "
            "Generate one concise, factual scene caption in English based on the keyframe descriptions.\n\n"
            f"Title: {video_title or 'Unknown'}\n"
            f"Category: {category or 'Unknown'}\n"
            f"Scene Index: {scene_index}\n"
            f"Scene Frame Range: {start_frame} - {end_frame}\n"
            f"Keyframe Descriptions:\n{keyframes_text}\n\n"
            "Return only the scene caption."
        )
        return call_llm(prompt=prompt, temperature=self.temperature)


class RuleBasedVideoCaptioner:
    """Deterministic local captioner used for offline/debug runs."""

    def __init__(self, temperature: float = 0.0) -> None:
        self.temperature = temperature

    def caption_scene(
        self,
        video_title: str | None,
        category: str | None,
        scene_index: int,
        start_frame: int,
        end_frame: int,
        keyframe_descriptions: Iterable[str],
    ) -> str:
        snippets = [str(item).strip() for item in keyframe_descriptions if str(item).strip()]
        concise = " ".join(snippets[:2]) if snippets else "visual activity changes across sampled frames"
        title_text = f" in {video_title}" if video_title else ""
        category_text = f" [{category}]" if category else ""
        return (
            f"Scene {scene_index}{category_text}{title_text} spans frames {start_frame}-{end_frame} and shows {concise}."
        )

    def aggregate_video_caption(
        self,
        video_title: str | None,
        category: str | None,
        scene_captions: List[str],
    ) -> str:
        if not scene_captions:
            return "No scene captions available."
        joined = " ".join(caption.strip() for caption in scene_captions[:5] if caption.strip())
        if video_title:
            return f"{video_title}: {joined}"
        return joined

    def aggregate_video_caption(
        self,
        video_title: str | None,
        category: str | None,
        scene_captions: List[str],
    ) -> str:
        print("Calling LLM for video caption aggregation...")
        scene_caption_text = "\n".join(f"Scene {index + 1}: {caption}" for index, caption in enumerate(scene_captions))
        prompt = (
            "You are a video summarization assistant. "
            "Aggregate the following scene captions into one coherent video-level caption in English.\n\n"
            f"Title: {video_title or 'Unknown'}\n"
            f"Category: {category or 'Unknown'}\n"
            f"Scene Captions:\n{scene_caption_text}\n\n"
            "Return only the aggregated video caption."
        )
        return call_llm(prompt=prompt, temperature=self.temperature)