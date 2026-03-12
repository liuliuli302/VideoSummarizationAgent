from __future__ import annotations

import os

from tqdm.auto import tqdm

from src.agents import EmotionAgent, InformationAgent, PlannerAgent, StoryAgent, VisualAgent
from src.caption import SegmentCaptioner
from src.data import InferenceResult, PlannerPlan, Segment, SegmentScore
from src.io import JsonSaver
from src.llm import build_llm_client
from src.memory.memory_manager import MemoryManager
from src.preprocessing.frame_mapper import FrameScoreMapper
from src.preprocessing.segmenter import build_segments_by_count, build_segments_by_frame_window
from src.preprocessing.video_reader import load_video_info


class VideoSummarizationPipeline:
    def __init__(
        self,
        llm_model: str,
        llm_mode: str = "api",
        output_root: str = "outputs/inference_results",
        enable_memory: bool = False,
        max_history_segments: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        llm_client = build_llm_client(
            llm_mode=llm_mode,
            model=llm_model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )
        self.output_root = output_root
        self.captioner = SegmentCaptioner(llm_client=llm_client)
        self.planner = PlannerAgent(llm_client=llm_client)
        self.experts = {
            "story_agent": StoryAgent(llm_client=llm_client),
            "visual_agent": VisualAgent(llm_client=llm_client),
            "emotion_agent": EmotionAgent(llm_client=llm_client),
            "information_agent": InformationAgent(llm_client=llm_client),
        }
        self.memory = MemoryManager(enabled=enable_memory, max_history_segments=max_history_segments)
        self.frame_mapper = FrameScoreMapper()
        self.json_saver = JsonSaver()

    def run(
        self,
        video_path: str,
        segment_mode: str = "count",
        segment_value: int = 8,
        segment_overlap: int = 0,
        caption_frames_per_segment: int = 5,
        original_picks: list[int] | None = None,
    ) -> InferenceResult:
        video_info = load_video_info(video_path)
        segments = self._build_segments(
            total_frames=video_info.total_frames,
            segment_mode=segment_mode,
            segment_value=segment_value,
            segment_overlap=segment_overlap,
            caption_frames_per_segment=caption_frames_per_segment,
        )

        captions = [
            self.captioner.caption_segment(video_path=video_path, segment=segment)
            for segment in tqdm(
                segments,
                total=len(segments),
                desc=f"Captioning {video_info.video_id}",
                leave=False,
            )
        ]
        planner_plan = self.planner.plan_video([item.caption for item in captions])
        segment_scores = self._score_segments(segments=segments, captions=captions, planner_plan=planner_plan)

        picks = original_picks if original_picks is not None else list(range(video_info.total_frames))
        frame_scores = self.frame_mapper.assign_segment_scores_to_original_frames(
            segment_scores=segment_scores,
            original_picks=picks,
            total_frames=video_info.total_frames,
        )

        output_dir = os.path.join(self.output_root, video_info.video_id)
        result = InferenceResult(
            video_id=video_info.video_id,
            video_info=video_info,
            planner_plan=planner_plan,
            captions=captions,
            segment_scores=segment_scores,
            frame_scores=frame_scores,
            frame_score_picks=picks,
            output_dir=output_dir,
        )
        self._save_outputs(result)
        return result

    def _build_segments(
        self,
        total_frames: int,
        segment_mode: str,
        segment_value: int,
        segment_overlap: int,
        caption_frames_per_segment: int,
    ) -> list[Segment]:
        if segment_mode == "count" and segment_overlap > 0:
            raise ValueError("segment_overlap is only supported for fixed_frames/sliding_window modes.")

        if segment_mode == "count":
            raw_segments = build_segments_by_count(
                total_frames=total_frames,
                num_segments=segment_value,
                caption_frames_per_segment=caption_frames_per_segment,
            )
        elif segment_mode in {"fixed_frames", "sliding_window"}:
            raw_segments = build_segments_by_frame_window(
                total_frames=total_frames,
                frames_per_segment=segment_value,
                caption_frames_per_segment=caption_frames_per_segment,
                overlap_frames=segment_overlap,
            )
        else:
            raise ValueError(f"Unsupported segment_mode: {segment_mode}")

        return [
            Segment(
                segment_id=int(item["segment_id"]),
                start_frame=int(item["start_frame"]),
                end_frame=int(item["end_frame"]),
                caption_frame_indices=list(item["caption_frame_indices"]),
            )
            for item in raw_segments
        ]

    def _score_segments(
        self,
        segments: list[Segment],
        captions,
        planner_plan: PlannerPlan,
    ) -> list[SegmentScore]:
        self.memory.reset()
        results: list[SegmentScore] = []

        segment_pairs = zip(segments, captions)
        for segment, caption in tqdm(
            segment_pairs,
            total=len(segments),
            desc="Scoring segments",
            leave=False,
        ):
            memory_context = self.memory.build_context()
            planner_score = self.planner.score_segment(
                planner_plan=planner_plan,
                current_caption=caption.caption,
                memory_context=memory_context,
            )
            expert_results = {
                name: agent.score_segment(
                    video_theme=planner_plan.video_theme,
                    global_summary=planner_plan.global_summary,
                    current_caption=caption.caption,
                    memory_context=memory_context,
                )
                for name, agent in self.experts.items()
            }

            final_score = planner_score.score + sum(
                planner_plan.expert_weights[name] * expert_results[name].score
                for name in planner_plan.expert_weights
            )

            results.append(
                SegmentScore(
                    segment_id=segment.segment_id,
                    start_frame=segment.start_frame,
                    end_frame=segment.end_frame,
                    planner_score=planner_score.score,
                    planner_reason=planner_score.reason,
                    expert_results=expert_results,
                    final_score=float(final_score),
                )
            )
            self.memory.append(caption.caption)

        return results

    def _save_outputs(self, result: InferenceResult) -> None:
        self.json_saver.save(
            os.path.join(result.output_dir, "captions.json"),
            {
                "video_id": result.video_id,
                "segments": [item.to_dict() for item in result.captions],
            },
        )
        self.json_saver.save(
            os.path.join(result.output_dir, "planner_plan.json"),
            {
                "video_id": result.video_id,
                **result.planner_plan.to_dict(),
            },
        )
        self.json_saver.save(
            os.path.join(result.output_dir, "segment_scores.json"),
            {
                "video_id": result.video_id,
                "segments": [item.to_dict() for item in result.segment_scores],
            },
        )
        self.json_saver.save(
            os.path.join(result.output_dir, "frame_scores.json"),
            {
                "video_id": result.video_id,
                "picks": result.frame_score_picks,
                "frame_scores": result.frame_scores,
            },
        )
        self.json_saver.save(os.path.join(result.output_dir, "inference_result.json"), result.to_dict())