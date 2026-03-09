"""Project command line entry for training, inference, evaluation and summary generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config import ConfigNode, load_config, set_nested_value
from src.datasets import BenchmarkDatasetAdapter
from src.evaluation import EvaluationBenchmark
from src.pipeline import VideoSummaryInferenceEngine
from src.solver import EvalSolver, ExperimentSolver, InferenceSolver


def load_optional_json(json_path: Optional[str]) -> Optional[Any]:
    """Load an optional JSON file used by summary/evaluation commands."""
    if not json_path:
        return None
    with open(json_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def build_parser() -> argparse.ArgumentParser:
    """Create the shared CLI parser used by all tasks."""
    parser = argparse.ArgumentParser(description="Video summarization research toolkit")
    parser.add_argument(
        "--task",
        type=str,
        required=False,
        default="summary",
        choices=["experiment", "inference", "eval", "summary"],
        help="Pipeline stage to run",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--dataset", type=str, default=None, help="Root directory of the real benchmark datasets")
    parser.add_argument("--dataset_name", type=str, default=None, help="Benchmark dataset name: SumMe or TVSum")
    parser.add_argument("--video_id", type=str, default=None, help="Video id to run on the selected benchmark dataset")
    parser.add_argument("--debug_video", type=str, default=None, help="Debug alias for selecting a single benchmark video")
    parser.add_argument("--video_path", type=str, default="data/raw/demo.mp4", help="Input video path")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for baseline training")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--title", type=str, default=None, help="Optional video title for summary inference")
    parser.add_argument("--category", type=str, default=None, help="Optional domain/category hint")
    parser.add_argument("--output_path", type=str, default=None, help="Optional output file path")
    parser.add_argument("--asr_path", type=str, default=None, help="Optional ASR JSON path for summary")
    parser.add_argument("--metadata_file", type=str, default=None, help="Optional CSV/JSON metadata file")
    return parser


def merge_cli_overrides(config: ConfigNode, args: argparse.Namespace) -> ConfigNode:
    """Merge lightweight CLI overrides into the loaded YAML configuration."""
    merged_config = config.copy()
    set_nested_value(merged_config, "experiment.epochs", args.epochs)
    set_nested_value(merged_config, "evaluation.num_samples", args.num_samples)

    if args.dataset:
        set_nested_value(merged_config, "dataset.data_root", args.dataset)
    if args.video_path:
        set_nested_value(merged_config, "dataset.video_path", args.video_path)
    if args.dataset_name:
        set_nested_value(merged_config, "dataset.name", args.dataset_name)

    if args.metadata_file:
        set_nested_value(merged_config, "dataset.metadata_file", args.metadata_file)
    return merged_config


def run_summary_task(config: ConfigNode, args: argparse.Namespace) -> dict[str, Any]:
    """Run the end-to-end summary engine and return the result payload."""
    engine = VideoSummaryInferenceEngine(config)
    return engine.run(
        video_path=args.video_path,
        title=args.title,
        category=args.category,
        asr_segments=load_optional_json(args.asr_path),
        output_path=args.output_path,
    )


def run_benchmark_pipeline(config: ConfigNode, args: argparse.Namespace) -> dict[str, Any]:
    """Run the full real-dataset pipeline on one benchmark video and evaluate it."""
    dataset_root = args.dataset or config.dataset.data_root
    if not args.dataset_name:
        raise ValueError("--dataset_name is required when running the real benchmark pipeline.")

    selected_video_id = args.debug_video or args.video_id
    if not selected_video_id:
        raise ValueError("--video_id or --debug_video is required when running the real benchmark pipeline.")

    dataset = BenchmarkDatasetAdapter(dataset_root=dataset_root, dataset_name=args.dataset_name)
    record = dataset.get_record(selected_video_id)

    runtime_config = config.copy()
    set_nested_value(runtime_config, "dataset.video_path", record.video_path)
    set_nested_value(runtime_config, "dataset.data_root", dataset_root)
    set_nested_value(runtime_config, "dataset.name", record.dataset_name)
    set_nested_value(runtime_config, "llm.backend", "rule_based")

    summary_output_path = args.output_path or os.path.join(
        runtime_config.paths.summary_output_dir,
        f"{record.dataset_name.lower()}_{record.video_id}_summary.json",
    )

    engine = VideoSummaryInferenceEngine(runtime_config)
    summary_result = engine.run(
        video_path=record.video_path,
        title=args.title or record.title,
        category=args.category or record.category,
        asr_segments=load_optional_json(args.asr_path),
        output_path=summary_output_path,
    )

    benchmark = EvaluationBenchmark()
    evaluation_report = benchmark.evaluate_dataset_record(
        record=record,
        predicted_scores=summary_result["frame_scores"],
        scene_ranges=summary_result["scene_ranges"],
        budget_ratio=float(runtime_config.summarization.budget_ratio),
    )

    evaluation_dir = os.path.join(runtime_config.evaluation.output_dir, record.dataset_name.lower(), record.video_id)
    os.makedirs(evaluation_dir, exist_ok=True)
    evaluation_path = os.path.join(evaluation_dir, f"{record.video_id}_evaluation.json")
    with open(evaluation_path, "w", encoding="utf-8") as file_obj:
        json.dump(evaluation_report, file_obj, ensure_ascii=False, indent=2)

    if evaluation_report.get("sampled_predicted_scores") and evaluation_report.get("sampled_gt_scores"):
        benchmark.save_report(
            metrics={
                **evaluation_report["metrics"],
                "sampled_score_alignment": evaluation_report["sampled_score_alignment"],
            },
            predicted_scores=evaluation_report["sampled_predicted_scores"],
            gt_scores=evaluation_report["sampled_gt_scores"],
            output_dir=evaluation_dir,
            video_id=record.video_id,
        )

    result = {
        "dataset_name": record.dataset_name,
        "video_id": record.video_id,
        "video_path": record.video_path,
        "summary_output_path": summary_result["output_path"],
        "evaluation_output_path": evaluation_path,
        "metrics": evaluation_report["metrics"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    config = merge_cli_overrides(load_config(args.config), args)

    print(f"Starting task: {args.task}")

    if args.dataset_name and (args.video_id or args.debug_video):
        run_benchmark_pipeline(config, args)
        return

    if args.task == "summary":
        result = run_summary_task(config, args)
        print(f"Summary saved to: {result['output_path']}")
        print(result["summary"])
        return

    if args.task == "experiment":
        solver = ExperimentSolver(config)
    elif args.task == "inference":
        solver = InferenceSolver(config)
    elif args.task == "eval":
        solver = EvalSolver(config)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    solver.run()


if __name__ == "__main__":
    main()
