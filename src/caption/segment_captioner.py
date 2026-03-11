from __future__ import annotations

from src.data.schemas import Segment, SegmentCaption
from src.llm.client import LLMClient
from src.llm.prompts import CAPTION_SYSTEM_PROMPT
from src.preprocessing.video_reader import read_frames


class SegmentCaptioner:
    def __init__(self, llm_client: LLMClient, fallback_text: str = "A short video segment.") -> None:
        self.llm_client = llm_client
        self.fallback_text = fallback_text

    def caption_segment(self, video_path: str, segment: Segment) -> SegmentCaption:
        images = read_frames(video_path=video_path, frame_indices=segment.caption_frame_indices)
        if not images:
            caption = self.fallback_text
        else:
            caption = self.llm_client.generate_multimodal_text(
                system_prompt=CAPTION_SYSTEM_PROMPT,
                user_prompt="Describe the sampled frames as one concise segment caption.",
                images=images,
            )

        return SegmentCaption(
            segment_id=segment.segment_id,
            start_frame=segment.start_frame,
            end_frame=segment.end_frame,
            caption_frame_indices=segment.caption_frame_indices,
            caption=caption.strip() or self.fallback_text,
        )