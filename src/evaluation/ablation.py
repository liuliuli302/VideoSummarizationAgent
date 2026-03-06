from __future__ import annotations

import copy
import json
import os
import time
from typing import Any, Dict, Iterable, Optional, Sequence

from src.evaluation.benchmark import EvaluationBenchmark
from src.pipeline import VideoSummaryInferenceEngine


DEFAULT_ABLATION_VARIANTS: Dict[str, Dict[str, Any]] = {
    "full_system": {},
    "no_planner": {"ablation": {"disable_planner": True}},
    "no_critic": {"ablation": {"disable_critic": True}},
    "no_domain": {"ablation": {"disable_domain": True}},
    "no_memory": {"ablation": {"disable_memory": True}},
    "compact_memory": {"memory": {"max_items_per_slot": 8, "similarity_prune_threshold": 0.7}},
}


class TrainingFreeAblationRunner:
    """Run configuration-only ablations without any parameter training."""

    def __init__(
        self,
        base_config: Optional[Dict[str, Any]] = None,
        benchmark: Optional[EvaluationBenchmark] = None,
    ) -> None:
        self.base_config = base_config or {}
        self.benchmark = benchmark or EvaluationBenchmark()

    def run(
        self,
        video_path: str,
        output_dir: str,
        title: Optional[str] = None,
        category: Optional[str] = None,
        asr_segments: Optional[Sequence[Dict[str, Any]]] = None,
        gt_scores: Optional[Sequence[int | float]] = None,
        variants: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        resolved_variants = variants or DEFAULT_ABLATION_VARIANTS

        report: Dict[str, Any] = {
            "video_path": video_path,
            "output_dir": output_dir,
            "variants": {},
        }

        for variant_name, overrides in resolved_variants.items():
            variant_dir = os.path.join(output_dir, variant_name)
            os.makedirs(variant_dir, exist_ok=True)
            variant_config = self._deep_merge(copy.deepcopy(self.base_config), overrides)
            engine = VideoSummaryInferenceEngine(config=variant_config)
            variant_output_path = os.path.join(variant_dir, f"{variant_name}_summary.json")

            start_time = time.perf_counter()
            result = engine.run(
                video_path=video_path,
                title=title,
                category=category,
                asr_segments=list(asr_segments) if asr_segments is not None else None,
                output_path=variant_output_path,
            )
            elapsed_sec = time.perf_counter() - start_time

            variant_report: Dict[str, Any] = {
                "output_path": result["output_path"],
                "elapsed_sec": round(elapsed_sec, 4),
                "selected_segment_count": len(result["selected_segments"]),
                "runtime_profile": result.get("runtime_profile", {}),
            }

            if gt_scores is not None:
                metrics = self.benchmark.evaluate(
                    predicted_scores=result["frame_scores"],
                    gt_scores=gt_scores,
                    selected_segments=result["selected_segments"],
                    latencies_sec=[elapsed_sec],
                )
                artifacts = self.benchmark.save_report(
                    metrics=metrics,
                    predicted_scores=result["frame_scores"],
                    gt_scores=gt_scores,
                    output_dir=variant_dir,
                    video_id=variant_name,
                )
                variant_report["metrics"] = metrics
                variant_report["artifacts"] = artifacts

            report["variants"][variant_name] = variant_report

        report_path = os.path.join(output_dir, "ablation_report.json")
        with open(report_path, "w", encoding="utf-8") as file_obj:
            json.dump(report, file_obj, ensure_ascii=False, indent=2)
        report["report_path"] = report_path
        return report

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base