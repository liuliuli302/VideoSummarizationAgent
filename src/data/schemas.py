from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any


@dataclass(slots=True)
class VideoInfo:
    video_id: str
    video_path: str
    fps: float
    total_frames: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Segment:
    segment_id: int
    start_frame: int
    end_frame: int
    caption_frame_indices: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SegmentCaption:
    segment_id: int
    start_frame: int
    end_frame: int
    caption_frame_indices: list[int]
    caption: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlannerPlan:
    video_theme: str
    global_summary: str
    expert_weights: dict[str, float]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentScore:
    score: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExpertResult:
    agent_name: str
    score: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SegmentScore:
    segment_id: int
    start_frame: int
    end_frame: int
    planner_score: float
    planner_reason: str
    expert_results: dict[str, ExpertResult]
    final_score: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expert_results"] = {
            name: result.to_dict() if hasattr(result, "to_dict") else result
            for name, result in self.expert_results.items()
        }
        return payload


@dataclass(slots=True)
class DatasetVideoRecord:
    dataset_name: str
    video_id: str
    video_path: str
    n_frames: int
    picks: list[int]
    change_points: list[list[int]]
    n_frame_per_seg: list[int]
    user_summary: list[list[float]]
    user_scores: list[list[float]] | list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InferenceResult:
    video_id: str
    video_info: VideoInfo
    planner_plan: PlannerPlan
    captions: list[SegmentCaption]
    segment_scores: list[SegmentScore]
    frame_scores: list[float]
    frame_score_picks: list[int]
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "video_info": self.video_info.to_dict(),
            "planner_plan": self.planner_plan.to_dict(),
            "captions": [item.to_dict() for item in self.captions],
            "segment_scores": [item.to_dict() for item in self.segment_scores],
            "frame_scores": self.frame_scores,
            "frame_score_picks": self.frame_score_picks,
            "output_dir": self.output_dir,
        }


@dataclass(slots=True)
class EvalResult:
    dataset_name: str
    video_id: str
    f1: float
    precision: float
    recall: float
    rho: float
    tau: float
    selected_summary: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvalVariantResult:
    variant_name: str
    frame_scores: list[float]
    eval_result: EvalResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_name": self.variant_name,
            "frame_scores": self.frame_scores,
            "eval_result": self.eval_result.to_dict(),
        }


@dataclass(slots=True)
class EvalOverview:
    exam_name: str
    total_videos: int
    per_variant_metrics: dict[str, dict[str, float]]
    per_dataset_metrics: dict[str, dict[str, dict[str, float]]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def build(exam_name: str, records: list[dict[str, Any]]) -> "EvalOverview":
        def _aggregate(items: list[dict[str, Any]]) -> dict[str, float]:
            if not items:
                return {
                    "count": 0.0,
                    "f1_mean": 0.0,
                    "precision_mean": 0.0,
                    "recall_mean": 0.0,
                    "rho_mean": 0.0,
                    "tau_mean": 0.0,
                }
            return {
                "count": float(len(items)),
                "f1_mean": float(mean(item["f1"] for item in items)),
                "precision_mean": float(mean(item["precision"] for item in items)),
                "recall_mean": float(mean(item["recall"] for item in items)),
                "rho_mean": float(mean(item["rho"] for item in items)),
                "tau_mean": float(mean(item["tau"] for item in items)),
            }

        per_variant: dict[str, list[dict[str, Any]]] = {}
        per_dataset: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for record in records:
            dataset_name = str(record["dataset_name"])
            variant_name = str(record["variant_name"])
            per_variant.setdefault(variant_name, []).append(record)
            per_dataset.setdefault(dataset_name, {}).setdefault(variant_name, []).append(record)

        return EvalOverview(
            exam_name=exam_name,
            total_videos=len({(str(item["dataset_name"]), str(item["video_id"])) for item in records}),
            per_variant_metrics={name: _aggregate(items) for name, items in per_variant.items()},
            per_dataset_metrics={
                dataset_name: {variant_name: _aggregate(items) for variant_name, items in variants.items()}
                for dataset_name, variants in per_dataset.items()
            },
        )