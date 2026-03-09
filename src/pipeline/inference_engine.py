from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from src.config import build_runtime_config
from src.datasets.schemas import Segment, Window
from src.preprocessing import build_fixed_segments, build_sliding_windows, load_video_meta
from src.preprocessing.window_builder import uniform_sample_indices
from src.pipeline.streaming_pipeline import StreamingVideoSummarizationPipeline
from src.scoring import BudgetedSummarySelector, FrameScoreFusion
from video_processing.shot_detection import detect_scenes


class VideoSummaryInferenceEngine:
    """End-to-end inference engine from video input to summary outputs."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        pipeline: Optional[StreamingVideoSummarizationPipeline] = None,
        frame_fusion: Optional[FrameScoreFusion] = None,
        selector: Optional[BudgetedSummarySelector] = None,
    ) -> None:
        self.config = build_runtime_config(config)
        selection_config = self.config.summarization.selection
        self.pipeline = pipeline or StreamingVideoSummarizationPipeline(config=self.config)
        self.frame_fusion = frame_fusion or FrameScoreFusion()
        self.selector = selector or BudgetedSummarySelector(
            frame_fusion=self.frame_fusion,
            min_label=selection_config.min_label,
            allow_partial_segment=bool(selection_config.allow_partial_segment),
        )

    def run(
        self,
        video_path: str,
        title: Optional[str] = None,
        category: Optional[str] = None,
        asr_segments: Optional[List[Dict[str, Any]]] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        video_meta = load_video_meta(
            video_path=video_path,
            title=title,
            category=category,
            asr_segments=asr_segments,
        )
        scene_ranges = detect_scenes(video_path)
        segments = self._build_scene_segments(video_meta=video_meta, scene_ranges=scene_ranges)
        windows = self._build_scene_windows(video_meta=video_meta, segments=segments)

        pipeline_result = self.pipeline.run(
            video_meta=video_meta,
            segments=segments,
            windows=windows,
            video_path=video_path,
        )

        frame_scores = self.frame_fusion.fuse(
            window_scores=self._deserialize_window_scores(pipeline_result["window_scores"]),
            windows=windows,
            total_frames=video_meta.total_frames,
        )
        selected_segments = self.selector.select(frame_scores=frame_scores, budget=self._budget_value())
        if not selected_segments:
            selected_segments = self._fallback_select(frame_scores=frame_scores)

        enriched_segments = self._enrich_selected_segments(
            selected_segments=selected_segments,
            windows=windows,
            window_scores=pipeline_result["window_scores"],
            window_features=pipeline_result["window_features"],
            fps=video_meta.fps,
        )
        summary_text = self._build_summary_text(enriched_segments)

        result = {
            "video_meta": video_meta.to_dict(),
            "scene_ranges": [{"start_frame": start, "end_frame": end} for start, end in scene_ranges],
            "segments": [segment.to_dict() for segment in segments],
            "windows": [window.to_dict() for window in windows],
            "global_context": pipeline_result["global_context"],
            "window_features": pipeline_result["window_features"],
            "window_scores": pipeline_result["window_scores"],
            "frame_scores": frame_scores,
            "selected_segments": enriched_segments,
            "summary": summary_text,
            "decision_logs": pipeline_result["decision_logs"],
            "final_memory": pipeline_result["final_memory"],
            "runtime_profile": pipeline_result.get("runtime_profile", {}),
        }

        save_path = output_path or self._default_output_path(video_meta["video_id"] if isinstance(video_meta, dict) else video_meta.video_id)
        self.save_results(result, save_path)
        result["output_path"] = save_path
        return result

    def _build_scene_segments(self, video_meta, scene_ranges: List[tuple[int, int]]) -> List[Segment]:
        if not scene_ranges:
            return build_fixed_segments(video_meta, self._segment_length_sec())

        segments: List[Segment] = []
        for index, (start_frame, end_frame) in enumerate(scene_ranges):
            safe_start = max(0, int(start_frame))
            safe_end = min(video_meta.total_frames, max(safe_start + 1, int(end_frame)))
            segments.append(
                Segment(
                    seg_id=f"scene_{index}",
                    start_frame=safe_start,
                    end_frame=safe_end,
                    start_sec=safe_start / video_meta.fps,
                    end_sec=safe_end / video_meta.fps,
                )
            )
        return segments

    def _build_scene_windows(self, video_meta, segments: List[Segment]) -> List[Window]:
        scene_sample_rate = int(self.config.summarization.window.sample_rate)
        windows: List[Window] = []
        for index, segment in enumerate(segments):
            windows.append(
                Window(
                    win_id=f"scene_w_{index}",
                    start_frame=segment.start_frame,
                    end_frame=segment.end_frame,
                    sampled_frame_indices=uniform_sample_indices(
                        segment.start_frame,
                        segment.end_frame,
                        scene_sample_rate,
                    ),
                )
            )
        return windows

    def save_results(self, result: Dict[str, Any], output_path: str) -> None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file_obj:
            json.dump(result, file_obj, ensure_ascii=False, indent=2)

    def _segment_length_sec(self) -> float:
        return float(self.config.summarization.segment_length_sec)

    def _window_config(self) -> Dict[str, Any]:
        return self.config.summarization.window.to_dict()

    def _budget_value(self) -> int | float:
        return self.config.summarization.budget_ratio

    def _default_output_path(self, video_id: str) -> str:
        output_dir = self.config.paths.summary_output_dir
        return os.path.join(output_dir, f"{video_id}_summary.json")

    def _deserialize_window_scores(self, serialized_scores: List[Dict[str, Any]]):
        from src.datasets.schemas import WindowScore

        return [WindowScore(**score_dict) for score_dict in serialized_scores]

    def _fallback_select(self, frame_scores: List[str]) -> List[Dict[str, Any]]:
        candidates = self.frame_fusion.extract_candidate_segments(frame_scores)
        if not candidates:
            return []
        best_segment = sorted(
            candidates,
            key=lambda item: (-item["priority"], -item["length"], item["start_frame"]),
        )[0]
        budget_frames = max(1, self.selector._resolve_budget_frames(self._budget_value(), len(frame_scores)))
        trimmed_end = min(best_segment["start_frame"] + budget_frames, best_segment["end_frame"])
        return [
            {
                **best_segment,
                "end_frame": trimmed_end,
                "length": trimmed_end - best_segment["start_frame"],
            }
        ]

    def _enrich_selected_segments(
        self,
        selected_segments: List[Dict[str, Any]],
        windows: List[Any],
        window_scores: List[Dict[str, Any]],
        window_features: List[Dict[str, Any]],
        fps: float,
    ) -> List[Dict[str, Any]]:
        enriched_segments: List[Dict[str, Any]] = []
        score_lookup = {item["win_id"]: item for item in window_scores}
        feature_lookup = {item["win_id"]: item for item in window_features}

        for index, segment in enumerate(selected_segments):
            supporting_windows = [
                window.win_id
                for window in windows
                if not (window.end_frame <= segment["start_frame"] or window.start_frame >= segment["end_frame"])
            ]
            reasons = []
            summaries = []
            for win_id in supporting_windows:
                score = score_lookup.get(win_id, {})
                feature = feature_lookup.get(win_id, {})
                if score.get("base_decision"):
                    reasons.append(score["base_decision"].splitlines()[1].replace("Reason:", "").strip())
                if feature.get("semantic_summary"):
                    summaries.append(feature["semantic_summary"].splitlines()[0].strip())

            unique_reasons = []
            for reason in reasons:
                if reason and reason not in unique_reasons:
                    unique_reasons.append(reason)

            unique_summaries = []
            for summary in summaries:
                if summary and summary not in unique_summaries:
                    unique_summaries.append(summary)

            enriched_segments.append(
                {
                    "segment_id": f"summary_{index}",
                    "start_frame": segment["start_frame"],
                    "end_frame": segment["end_frame"],
                    "start_sec": round(segment["start_frame"] / fps, 3),
                    "end_sec": round(segment["end_frame"] / fps, 3),
                    "label": segment["label"],
                    "length": segment["length"],
                    "supporting_windows": supporting_windows,
                    "summary_text": " ".join(unique_summaries[:2]),
                    "selection_reason": "；".join(unique_reasons[:2]) or "基于窗口重要性与预算约束选入摘要。",
                }
            )

        return enriched_segments

    def _build_summary_text(self, selected_segments: List[Dict[str, Any]]) -> str:
        if not selected_segments:
            return "No segments selected under the current budget."
        ordered_texts = []
        for index, segment in enumerate(selected_segments, start=1):
            summary_text = segment.get("summary_text") or segment.get("selection_reason") or ""
            ordered_texts.append(f"[{index}] {summary_text}")
        return "\n".join(ordered_texts)