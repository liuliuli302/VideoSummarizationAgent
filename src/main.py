"""Project command line entry for training, inference, evaluation and summary generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.pipeline import VideoSummaryInferenceEngine
from src.solver import EvalSolver, ExperimentSolver, InferenceSolver


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file.

    Missing configs are treated as an empty override to keep CLI usage simple for
    quick experiments.
    """
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}, using defaults.")
        return {}

    with open(config_path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj) or {}


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
        required=True,
        choices=["experiment", "inference", "eval", "summary"],
        help="Pipeline stage to run",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--video_path", type=str, default="data/raw/demo.mp4", help="Input video path")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for baseline training")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--title", type=str, default=None, help="Optional video title for summary inference")
    parser.add_argument("--category", type=str, default=None, help="Optional domain/category hint")
    parser.add_argument("--output_path", type=str, default=None, help="Optional output file path")
    parser.add_argument("--asr_path", type=str, default=None, help="Optional ASR JSON path for summary")
    parser.add_argument("--metadata_file", type=str, default=None, help="Optional CSV/JSON metadata file")
    return parser


def merge_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge lightweight CLI overrides into the loaded YAML configuration."""
    merged_config = dict(config)
    merged_config["video_path"] = args.video_path
    merged_config["epochs"] = args.epochs
    merged_config["num_samples"] = args.num_samples

    if args.metadata_file:
        merged_config["metadata_file"] = args.metadata_file
    return merged_config


def run_summary_task(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run the end-to-end summary engine and return the result payload."""
    engine = VideoSummaryInferenceEngine(config)
    return engine.run(
        video_path=args.video_path,
        title=args.title,
        category=args.category,
        asr_segments=load_optional_json(args.asr_path),
        output_path=args.output_path,
    )


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    config = merge_cli_overrides(load_config(args.config), args)

    print(f"Starting task: {args.task}")

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
