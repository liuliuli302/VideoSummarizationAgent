from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.agents import (
    CounterfactualCritic,
    DomainAgent,
    EventAgent,
    MainlineAgent,
    NoveltyAgent,
    PlannerAgent,
    TemporalAgent,
)
from src.datasets.schemas import MemoryState, PlannerOutput, VideoMeta, Window, WindowFeature, WindowScore
from src.memory import MemoryBank
from src.perception import RuleBasedCaptioner, WindowFeatureBuilder, WindowTextEncoder, WindowVisualEncoder
from src.pipeline.global_pipeline import GlobalUnderstandingPipeline
from src.scoring import AggregationAgent


class StreamingVideoSummarizationPipeline:
    """Offline-streaming pipeline that processes windows sequentially."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        global_pipeline: Optional[GlobalUnderstandingPipeline] = None,
        visual_encoder: Optional[WindowVisualEncoder] = None,
        captioner: Optional[RuleBasedCaptioner] = None,
        text_encoder: Optional[WindowTextEncoder] = None,
        feature_builder: Optional[WindowFeatureBuilder] = None,
        memory_bank: Optional[MemoryBank] = None,
        planner_agent: Optional[PlannerAgent] = None,
        mainline_agent: Optional[MainlineAgent] = None,
        novelty_agent: Optional[NoveltyAgent] = None,
        event_agent: Optional[EventAgent] = None,
        temporal_agent: Optional[TemporalAgent] = None,
        domain_agent: Optional[DomainAgent] = None,
        counterfactual_critic: Optional[CounterfactualCritic] = None,
        aggregation_agent: Optional[AggregationAgent] = None,
    ) -> None:
        self.config = config or {}
        self.ablation_config = self.config.get("ablation", {})
        self.optimization_config = self.config.get("optimization", {})
        memory_config = self.config.get("memory", {})
        aggregation_config = self.optimization_config.get("aggregation", {})

        self.global_pipeline = global_pipeline or GlobalUnderstandingPipeline()
        self.visual_encoder = visual_encoder or WindowVisualEncoder()
        self.captioner = captioner or RuleBasedCaptioner(visual_encoder=self.visual_encoder)
        self.text_encoder = text_encoder or WindowTextEncoder()
        self.feature_builder = feature_builder or WindowFeatureBuilder()
        self.memory_bank = memory_bank or MemoryBank(
            topk=int(memory_config.get("topk", 5)),
            max_items_per_slot=int(memory_config.get("max_items_per_slot", 50)),
            similarity_prune_threshold=float(memory_config.get("similarity_prune_threshold", 0.9)),
        )
        self.planner_agent = planner_agent or PlannerAgent()
        self.mainline_agent = mainline_agent or MainlineAgent()
        self.novelty_agent = novelty_agent or NoveltyAgent()
        self.event_agent = event_agent or EventAgent()
        self.temporal_agent = temporal_agent or TemporalAgent()
        self.domain_agent = domain_agent or DomainAgent()
        self.counterfactual_critic = counterfactual_critic or CounterfactualCritic()
        self.aggregation_agent = aggregation_agent or AggregationAgent(
            score_weights=aggregation_config.get("score_weights"),
            decision_thresholds=aggregation_config.get("decision_thresholds"),
        )

    def run(
        self,
        video_meta: VideoMeta,
        segments: List[Any],
        windows: List[Window],
        video_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.ablation_config.get("disable_memory", False):
            self.memory_bank.reset()
        resolved_video_path = video_path or video_meta.file_path
        global_context = self.global_pipeline.build_global_context(
            video_meta=video_meta,
            segments=segments,
            video_path=resolved_video_path,
        )
        window_features = self._build_window_features(
            video_meta=video_meta,
            windows=windows,
            video_path=resolved_video_path,
        )

        window_scores: List[WindowScore] = []
        decision_logs: List[Dict[str, Any]] = []

        for index, window in enumerate(windows):
            window_feature = window_features[index]
            memory_snapshot = self._memory_snapshot()
            memory_read = self._memory_read(window_feature.semantic_summary)
            planner_output = self._build_planner_output(
                window_feature=window_feature,
                summary_goal=global_context["summary_goal"],
                memory_state=memory_snapshot,
            )

            next_summary = ""
            if index + 1 < len(window_features):
                next_summary = window_features[index + 1].semantic_summary
            prev_summary = memory_snapshot.temporal_context[-1] if memory_snapshot.temporal_context else ""
            asr_text = self._normalize_asr(
                self._select_asr_for_window(video_meta=video_meta, window=window)
            )

            expert_outputs = {
                "mainline": self.mainline_agent.run(
                    window_summary=window_feature.semantic_summary,
                    summary_goal=global_context["summary_goal"],
                    story_memory=memory_read["story_ctx"],
                ),
                "novelty": self.novelty_agent.run(
                    window_summary=window_feature.semantic_summary,
                    selected_memory=memory_read["selected_ctx"],
                    recent_context=memory_read["temporal_ctx"],
                ),
                "event": self.event_agent.run(
                    visual_description=window_feature.visual_description,
                    local_caption=window_feature.local_caption,
                    asr_text=asr_text,
                ),
                "temporal": self.temporal_agent.run(
                    prev_summary=prev_summary,
                    current_summary=window_feature.semantic_summary,
                    next_summary=next_summary,
                    summary_chain=memory_snapshot.temporal_context,
                ),
                "domain": self._build_domain_output(
                    video_theme=self._dominant_theme(global_context.get("theme_dist", {})),
                    domain_hint=video_meta.category or "general",
                    window_summary=window_feature.semantic_summary,
                ),
            }
            critic_output = self._build_critic_output(
                memory_state=memory_snapshot,
                window_summary=window_feature.semantic_summary,
                summary_goal=global_context["summary_goal"],
            )
            window_score = self.aggregation_agent.run(
                win_id=window.win_id,
                expert_outputs=expert_outputs,
                planner_plan=planner_output.route_plan_text,
                critic_output=critic_output,
            )

            is_selected = window_score.final_importance in {"必须保留", "建议保留"}
            updated_memory = self._memory_update(
                window_feature=window_feature,
                final_decision=window_score.final_importance,
                is_selected=is_selected,
            )

            window_scores.append(window_score)
            decision_logs.append(
                {
                    "win_id": window.win_id,
                    "window": window.to_dict(),
                    "feature": window_feature.to_dict(),
                    "memory_before": memory_snapshot.to_dict(),
                    "memory_read": memory_read,
                    "planner_output": planner_output.to_dict(),
                    "expert_outputs": expert_outputs,
                    "critic_output": critic_output,
                    "window_score": window_score.to_dict(),
                    "memory_after": updated_memory.to_dict(),
                }
            )

        return {
            "global_context": global_context,
            "window_features": [feature.to_dict() for feature in window_features],
            "window_scores": [score.to_dict() for score in window_scores],
            "decision_logs": decision_logs,
            "final_memory": self._memory_snapshot().to_dict(),
            "runtime_profile": {
                "ablation": dict(self.ablation_config),
                "optimization": dict(self.optimization_config),
            },
        }

    def _memory_snapshot(self) -> MemoryState:
        if self.ablation_config.get("disable_memory", False):
            return MemoryState()
        return self.memory_bank.snapshot()

    def _memory_read(self, current_summary: str) -> Dict[str, list[str]]:
        if self.ablation_config.get("disable_memory", False):
            return {"selected_ctx": [], "story_ctx": [], "temporal_ctx": []}
        return self.memory_bank.read(current_summary)

    def _memory_update(
        self,
        window_feature: WindowFeature,
        final_decision: str,
        is_selected: bool,
    ) -> MemoryState:
        if self.ablation_config.get("disable_memory", False):
            return MemoryState()
        return self.memory_bank.update(
            window_feature=window_feature,
            final_decision=final_decision,
            is_selected=is_selected,
        )

    def _build_planner_output(
        self,
        window_feature: WindowFeature,
        summary_goal: str,
        memory_state: MemoryState,
    ) -> PlannerOutput:
        if self.ablation_config.get("disable_planner", False):
            return PlannerOutput(
                route_plan_text="Priority Experts: MainlineAgent -> NoveltyAgent -> EventAgent",
                focus_points=["使用静态专家顺序作为消融对照"],
                routing_rationale="Planner 已关闭，使用固定专家审查顺序。",
            )
        return self.planner_agent.run(
            window_context_text=window_feature.semantic_summary,
            summary_goal_text=summary_goal,
            memory_state=memory_state,
        )

    def _build_domain_output(self, video_theme: str, domain_hint: str, window_summary: str) -> str:
        if self.ablation_config.get("disable_domain", False):
            return (
                "Domain Match: 低\n"
                "Policy Suggestion: 领域专家已关闭，避免提供领域加成。\n"
                "Reason: 当前运行配置执行无领域专家消融。"
            )
        return self.domain_agent.run(
            video_theme=video_theme,
            domain_hint=domain_hint,
            window_summary=window_summary,
        )

    def _build_critic_output(self, memory_state: MemoryState, window_summary: str, summary_goal: str) -> str:
        if self.ablation_config.get("disable_critic", False):
            return (
                "Marginal Contribution: 中\n"
                "Loss If Removed: Counterfactual Critic 已关闭，未提供额外边际贡献校准。\n"
                "Recommendation: 视预算决定"
            )
        return self.counterfactual_critic.run(
            memory_summary=memory_state,
            window_summary=window_summary,
            summary_goal=summary_goal,
        )

    def _build_window_features(
        self,
        video_meta: VideoMeta,
        windows: List[Window],
        video_path: str,
    ) -> List[WindowFeature]:
        features: List[WindowFeature] = []
        for window in windows:
            asr_for_window = self._select_asr_for_window(video_meta=video_meta, window=window)
            visual_description = self.visual_encoder.describe_window(window=window, video_path=video_path)
            local_caption = self.captioner.generate_caption(window=window, video_path=video_path)
            semantic_summary = self.text_encoder.build_semantic_summary(
                local_caption=local_caption,
                title=video_meta.title,
                asr_text=asr_for_window,
            )
            features.append(
                self.feature_builder.build(
                    window=window,
                    visual_description=visual_description,
                    local_caption=local_caption,
                    semantic_summary=semantic_summary,
                    asr_text=asr_for_window,
                )
            )
        return features

    def _select_asr_for_window(self, video_meta: VideoMeta, window: Window) -> Optional[List[Dict[str, Any]]]:
        if not video_meta.asr_segments:
            return None

        start_sec = window.start_frame / video_meta.fps
        end_sec = window.end_frame / video_meta.fps
        matched_segments: List[Dict[str, Any]] = []
        for asr_segment in video_meta.asr_segments:
            if not isinstance(asr_segment, dict):
                continue
            start = asr_segment.get("start_sec", asr_segment.get("start", asr_segment.get("begin")))
            end = asr_segment.get("end_sec", asr_segment.get("end", asr_segment.get("finish")))
            if start is None and end is None:
                matched_segments.append(asr_segment)
                continue

            start_value = float(start if start is not None else start_sec)
            end_value = float(end if end is not None else end_sec)
            overlaps = not (end_value < start_sec or start_value > end_sec)
            if overlaps:
                matched_segments.append(asr_segment)

        return matched_segments or None

    def _normalize_asr(self, asr_segments: Optional[List[Dict[str, Any]]]) -> str:
        if not asr_segments:
            return ""

        parts = []
        for segment in asr_segments:
            text = segment.get("text") or segment.get("utterance") or segment.get("content")
            if text:
                parts.append(" ".join(str(text).strip().split()))
        return " ".join(parts)

    def _dominant_theme(self, theme_dist: Dict[str, float]) -> str:
        if not theme_dist:
            return "general"
        return max(theme_dist, key=theme_dist.get)