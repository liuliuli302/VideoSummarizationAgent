"""Minimal CLI for the LLM-agent video summarization pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data import DatasetLoader, EvalVariantResult
from src.evaluation import EvaluationReporter, SplitEvaluationAggregator, VsumEvaluator
from src.io import JsonSaver
from src.pipeline import VideoSummarizationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal video summarization agent system")
    parser.add_argument(
        "--task",
        type=str,
        required=False,
        default="run",
        choices=["run", "eval", "run_eval"],
        help="Run inference, evaluation, or both",
    )
    parser.add_argument("--video_path", type=str, default=None, help="Input video path")
    parser.add_argument("--dataset_root", type=str, default="/root/autodl-tmp/datasets", help="Dataset root path")
    parser.add_argument("--dataset_name", type=str, default=None, help="summe or tvsum")
    parser.add_argument("--video_id", type=str, default=None, help="Benchmark video id or mapped video name")
    parser.add_argument("--llm_mode", type=str, default="api", choices=["api", "mock"], help="Use real API or deterministic mock client")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--segment_mode", type=str, default="count", choices=["count", "fixed_frames", "sliding_window"])
    parser.add_argument("--segment_value", type=int, default=8, help="Segment count or frames per segment")
    parser.add_argument("--segment_overlap", type=int, default=0, help="Overlap in frames for fixed_frames/sliding_window modes")
    parser.add_argument("--caption_frames_per_segment", type=int, default=4)
    parser.add_argument("--enable_memory", action="store_true")
    parser.add_argument("--max_history_segments", type=int, default=None)
    parser.add_argument("--output_root", type=str, default="outputs/inference_results")
    parser.add_argument("--eval_output_root", type=str, default="outputs/evaluation")
    parser.add_argument("--eval_experiment_name", type=str, default=None)
    parser.add_argument("--eval_smooth_window", type=int, default=5)
    parser.add_argument("--split_root", type=str, default="/root/autodl-tmp/datasets/splits")
    parser.add_argument("--split_count", type=int, default=5, choices=[5, 50])
    return parser


def resolve_inputs(args: argparse.Namespace) -> tuple[str, list[int] | None, Any | None]:
    if args.dataset_name and args.video_id:
        record = DatasetLoader(args.dataset_root).load_record(args.dataset_name, args.video_id)
        return record.video_path, record.picks, record
    if not args.video_path:
        raise ValueError("Either --video_path or (--dataset_name and --video_id) is required.")
    return args.video_path, None, None


def evaluate_record(
    pipeline: VideoSummarizationPipeline,
    evaluator: VsumEvaluator,
    reporter: EvaluationReporter,
    record,
    args: argparse.Namespace,
    inference_result=None,
):
    if inference_result is None:
        inference_result = pipeline.run(
            video_path=record.video_path,
            segment_mode=args.segment_mode,
            segment_value=args.segment_value,
            segment_overlap=args.segment_overlap,
            caption_frames_per_segment=args.caption_frames_per_segment,
            original_picks=record.picks,
        )
    normalized_scores = reporter.normalize_scores(inference_result.frame_scores)
    smoothed_scores = reporter.smooth_scores(normalized_scores, window_size=args.eval_smooth_window)

    variants = [
        EvalVariantResult(
            variant_name="normalized_raw",
            frame_scores=normalized_scores,
            eval_result=evaluator.evaluate(frame_scores=normalized_scores, record=record),
        ),
        EvalVariantResult(
            variant_name="normalized_smoothed",
            frame_scores=smoothed_scores,
            eval_result=evaluator.evaluate(frame_scores=smoothed_scores, record=record),
        ),
    ]
    artifacts = reporter.save_video_report(
        record=record,
        variants=variants,
        original_frame_scores=inference_result.frame_scores,
        smooth_window_size=args.eval_smooth_window,
    )
    eval_path = os.path.join(
        artifacts["video_dir"],
        f"eval_{Path(record.video_id).stem}.json",
    )
    JsonSaver().save(
        eval_path,
        {
            "exam_name": reporter.exam_name,
            "dataset_name": record.dataset_name,
            "video_id": record.video_id,
            "variants": [variant.to_dict() for variant in variants],
        },
    )
    return inference_result, variants, artifacts


def run_dataset_split_evaluation(args: argparse.Namespace) -> None:
    if not args.dataset_name:
        raise ValueError("Dataset split evaluation requires --dataset_name.")

    dataset_loader = DatasetLoader(args.dataset_root)
    video_ids = dataset_loader.list_video_ids(args.dataset_name)
    if not video_ids:
        raise ValueError(f"No videos found for dataset={args.dataset_name}.")

    pipeline = VideoSummarizationPipeline(
        llm_model=args.llm_model,
        llm_mode=args.llm_mode,
        output_root=args.output_root,
        enable_memory=bool(args.enable_memory),
        max_history_segments=args.max_history_segments,
    )
    evaluator = VsumEvaluator()
    reporter = EvaluationReporter(
        output_root=args.eval_output_root,
        exam_name=args.eval_experiment_name,
    )

    total = len(video_ids)
    for index, video_id in enumerate(video_ids, start=1):
        record = dataset_loader.load_record(args.dataset_name, video_id)
        _, _, artifacts = evaluate_record(
            pipeline=pipeline,
            evaluator=evaluator,
            reporter=reporter,
            record=record,
            args=args,
        )
        print(f"[{index}/{total}] Evaluation saved to: {artifacts['video_dir']}")

    split_aggregator = SplitEvaluationAggregator(
        split_root=args.split_root,
        split_count=args.split_count,
    )
    split_summary = split_aggregator.aggregate_exam(
        exam_dir=reporter.exam_dir,
        datasets=[args.dataset_name.lower()],
    )
    print(f"Split overview updated at: {os.path.join(reporter.exam_dir, 'split_overview.json')}")
    print(f"Processed videos: {total}")
    print(f"Split count: {split_summary['split_count']}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.dataset_name and not args.video_id and args.task in {"eval", "run_eval"}:
        run_dataset_split_evaluation(args)
        return

    video_path, picks, record = resolve_inputs(args)

    pipeline = VideoSummarizationPipeline(
        llm_model=args.llm_model,
        llm_mode=args.llm_mode,
        output_root=args.output_root,
        enable_memory=bool(args.enable_memory),
        max_history_segments=args.max_history_segments,
    )

    if args.task in {"run", "run_eval"}:
        inference_result = pipeline.run(
            video_path=video_path,
            segment_mode=args.segment_mode,
            segment_value=args.segment_value,
            segment_overlap=args.segment_overlap,
            caption_frames_per_segment=args.caption_frames_per_segment,
            original_picks=picks,
        )
        print(f"Inference saved to: {inference_result.output_dir}")

    if args.task in {"eval", "run_eval"}:
        if record is None:
            raise ValueError("Evaluation requires --dataset_name and --video_id.")
        evaluator = VsumEvaluator()
        reporter = EvaluationReporter(
            output_root=args.eval_output_root,
            exam_name=args.eval_experiment_name,
        )
        if args.task == "eval":
            inference_result = pipeline.run(
                video_path=video_path,
                segment_mode=args.segment_mode,
                segment_value=args.segment_value,
                segment_overlap=args.segment_overlap,
                caption_frames_per_segment=args.caption_frames_per_segment,
                original_picks=picks,
            )

        _, _, artifacts = evaluate_record(
            pipeline=pipeline,
            evaluator=evaluator,
            reporter=reporter,
            record=record,
            args=args,
            inference_result=inference_result,
        )
        print(f"Evaluation saved to: {artifacts['video_dir']}")
        print(f"Overview updated at: {artifacts['overview_path']}")


if __name__ == "__main__":
    main()
