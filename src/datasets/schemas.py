from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def _validate_non_negative(value: float | int, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value}.")


def _validate_frame_range(start_frame: int, end_frame: int, field_name: str) -> None:
    _validate_non_negative(start_frame, f"{field_name}.start_frame")
    _validate_non_negative(end_frame, f"{field_name}.end_frame")
    if end_frame < start_frame:
        raise ValueError(
            f"{field_name}.end_frame must be >= start_frame, got {start_frame} -> {end_frame}."
        )


def _validate_time_range(start_sec: float, end_sec: float, field_name: str) -> None:
    _validate_non_negative(start_sec, f"{field_name}.start_sec")
    _validate_non_negative(end_sec, f"{field_name}.end_sec")
    if end_sec < start_sec:
        raise ValueError(
            f"{field_name}.end_sec must be >= start_sec, got {start_sec} -> {end_sec}."
        )


@dataclass(slots=True)
class VideoMeta:
    video_id: str
    file_path: str
    fps: float
    total_frames: int
    duration_sec: float
    title: Optional[str] = None
    asr_segments: Optional[List[Dict[str, Any]]] = None
    category: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.video_id:
            raise ValueError("video_id must not be empty.")
        if not self.file_path:
            raise ValueError("file_path must not be empty.")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}.")
        _validate_non_negative(self.total_frames, "total_frames")
        _validate_non_negative(self.duration_sec, "duration_sec")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Segment:
    seg_id: str
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float

    def __post_init__(self) -> None:
        if not self.seg_id:
            raise ValueError("seg_id must not be empty.")
        _validate_frame_range(self.start_frame, self.end_frame, "Segment")
        _validate_time_range(self.start_sec, self.end_sec, "Segment")

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Window:
    win_id: str
    start_frame: int
    end_frame: int
    sampled_frame_indices: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.win_id:
            raise ValueError("win_id must not be empty.")
        _validate_frame_range(self.start_frame, self.end_frame, "Window")
        for frame_idx in self.sampled_frame_indices:
            _validate_non_negative(frame_idx, "sampled_frame_indices item")
            if frame_idx < self.start_frame or frame_idx > self.end_frame:
                raise ValueError(
                    "sampled_frame_indices item must fall inside the window frame range. "
                    f"Got {frame_idx} outside [{self.start_frame}, {self.end_frame}]."
                )

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WindowFeature:
    win_id: str
    visual_description: str
    local_caption: str
    semantic_summary: str
    evidence_notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.win_id:
            raise ValueError("win_id must not be empty.")
        if not self.visual_description.strip():
            raise ValueError("visual_description must not be empty.")
        if not self.local_caption.strip():
            raise ValueError("local_caption must not be empty.")
        if not self.semantic_summary.strip():
            raise ValueError("semantic_summary must not be empty.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MemoryState:
    story_slots: List[str] = field(default_factory=list)
    selected_slots: List[str] = field(default_factory=list)
    redundancy_bank: List[str] = field(default_factory=list)
    temporal_context: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlannerOutput:
    route_plan_text: str
    focus_points: List[str] = field(default_factory=list)
    routing_rationale: str = ""

    def __post_init__(self) -> None:
        if not self.route_plan_text.strip():
            raise ValueError("route_plan_text must not be empty.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WindowScore:
    win_id: str
    expert_opinions: Dict[str, str] = field(default_factory=dict)
    base_decision: str = ""
    cf_comment: str = ""
    final_importance: str = ""

    def __post_init__(self) -> None:
        if not self.win_id:
            raise ValueError("win_id must not be empty.")
        if not self.base_decision.strip():
            raise ValueError("base_decision must not be empty.")
        if not self.final_importance.strip():
            raise ValueError("final_importance must not be empty.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)