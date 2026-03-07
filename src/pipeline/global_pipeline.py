from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.config import build_runtime_config
from src.datasets.schemas import Segment, VideoMeta, Window
from src.perception.captioner import RuleBasedCaptioner
from src.perception.text_encoder import WindowTextEncoder
from src.preprocessing.window_builder import uniform_sample_indices


class GlobalUnderstandingPipeline:
    """Generate coarse global understanding from fixed video segments."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        visual_encoder=None,
        captioner: Optional[RuleBasedCaptioner] = None,
        text_encoder: Optional[WindowTextEncoder] = None,
        sample_rate: Optional[int] = None,
    ) -> None:
        self.config = build_runtime_config(config)
        resolved_sample_rate = int(sample_rate or self.config.video.global_sample_rate)
        if resolved_sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {resolved_sample_rate}.")

        self.captioner = captioner or RuleBasedCaptioner(visual_encoder=visual_encoder)
        self.text_encoder = text_encoder or WindowTextEncoder()
        self.sample_rate = resolved_sample_rate

    def build_global_context(
        self,
        video_meta: VideoMeta,
        segments: List[Segment],
        video_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not segments:
            theme_dist = self._infer_theme_distribution(video_meta=video_meta, global_story="")
            return {
                "global_captions": [],
                "global_story": "No coarse segments available for global understanding.",
                "theme_dist": theme_dist,
                "summary_goal": self._build_summary_goal(video_meta, "", theme_dist),
            }

        resolved_video_path = video_path or video_meta.file_path
        global_captions: List[str] = []

        for segment in segments:
            segment_caption = self._generate_segment_caption(
                video_meta=video_meta,
                segment=segment,
                video_path=resolved_video_path,
            )
            global_captions.append(segment_caption)

        global_story = self._aggregate_global_story(global_captions)
        theme_dist = self._infer_theme_distribution(video_meta=video_meta, global_story=global_story)
        summary_goal = self._build_summary_goal(video_meta, global_story, theme_dist)

        return {
            "global_captions": global_captions,
            "global_story": global_story,
            "theme_dist": theme_dist,
            "summary_goal": summary_goal,
        }

    def _generate_segment_caption(
        self,
        video_meta: VideoMeta,
        segment: Segment,
        video_path: str,
    ) -> str:
        segment_window = Window(
            win_id=segment.seg_id,
            start_frame=segment.start_frame,
            end_frame=segment.end_frame,
            sampled_frame_indices=uniform_sample_indices(
                segment.start_frame,
                segment.end_frame,
                self.sample_rate,
            ),
        )
        local_caption = self.captioner.generate_caption(window=segment_window, video_path=video_path)
        segment_asr = self._select_asr_for_segment(video_meta.asr_segments, segment)
        semantic_summary = self.text_encoder.build_semantic_summary(
            local_caption=local_caption,
            title=video_meta.title,
            asr_text=segment_asr,
        )
        return f"{segment.seg_id}: {semantic_summary}"

    def _select_asr_for_segment(
        self,
        asr_segments: Optional[List[Dict[str, Any]]],
        segment: Segment,
    ) -> Optional[List[Dict[str, Any]]]:
        if not asr_segments:
            return None

        matched_segments: List[Dict[str, Any]] = []
        for asr_segment in asr_segments:
            if not isinstance(asr_segment, dict):
                continue

            start = asr_segment.get("start_sec", asr_segment.get("start", asr_segment.get("begin")))
            end = asr_segment.get("end_sec", asr_segment.get("end", asr_segment.get("finish")))
            if start is None and end is None:
                matched_segments.append(asr_segment)
                continue

            start_sec = float(start if start is not None else segment.start_sec)
            end_sec = float(end if end is not None else segment.end_sec)
            overlaps = not (end_sec < segment.start_sec or start_sec > segment.end_sec)
            if overlaps:
                matched_segments.append(asr_segment)

        return matched_segments or None

    def _aggregate_global_story(self, global_captions: List[str]) -> str:
        ordered_segments = []
        for index, caption in enumerate(global_captions, start=1):
            ordered_segments.append(f"Segment {index}: {caption}")
        return " ".join(ordered_segments)

    def _infer_theme_distribution(self, video_meta: VideoMeta, global_story: str) -> Dict[str, float]:
        text_blob = " ".join(
            part for part in [video_meta.title or "", video_meta.category or "", global_story] if part
        ).lower()

        theme_rules = {
            "travel": ["travel", "trip", "city", "street", "journey", "vlog"],
            "sports": ["sport", "match", "team", "goal", "game", "race"],
            "education": ["lesson", "tutorial", "class", "teach", "explain", "lecture"],
            "narrative": ["story", "character", "scene", "dialogue", "episode", "plot"],
        }

        scores = {theme: 1.0 for theme in theme_rules}
        scores["general"] = 1.0
        for theme, keywords in theme_rules.items():
            for keyword in keywords:
                if keyword in text_blob:
                    scores[theme] += 1.0

        total_score = sum(scores.values())
        return {theme: round(score / total_score, 4) for theme, score in scores.items()}

    def _build_summary_goal(
        self,
        video_meta: VideoMeta,
        global_story: str,
        theme_dist: Dict[str, float],
    ) -> str:
        dominant_theme = max(theme_dist, key=theme_dist.get) if theme_dist else "general"
        title_part = f"Title: {video_meta.title}. " if video_meta.title else ""
        category_part = f"Category: {video_meta.category}. " if video_meta.category else ""
        story_part = global_story if global_story else "No global story available."

        return (
            f"{title_part}{category_part}Primary theme: {dominant_theme}. "
            f"Summary goal: preserve the main progression across coarse segments and retain the most representative events. "
            f"Global story: {story_part}"
        )