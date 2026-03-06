import argparse
import json
import os
import sys

import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation import TrainingFreeAblationRunner


def load_yaml(path: str):
    with open(path, 'r', encoding='utf-8') as file_obj:
        return yaml.safe_load(file_obj) or {}


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as file_obj:
        return json.load(file_obj)


def main():
    parser = argparse.ArgumentParser(description="Run training-free ablations for video summarization.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Base config path")
    parser.add_argument("--video_path", type=str, required=True, help="Video path")
    parser.add_argument("--output_dir", type=str, default="outputs/ablation", help="Output directory")
    parser.add_argument("--title", type=str, default=None, help="Optional title")
    parser.add_argument("--category", type=str, default=None, help="Optional category")
    parser.add_argument("--asr_path", type=str, default=None, help="Optional ASR json path")
    parser.add_argument("--gt_scores_path", type=str, default=None, help="Optional GT scores json path")
    args = parser.parse_args()

    config = load_yaml(args.config)
    asr_segments = load_json(args.asr_path) if args.asr_path else None
    gt_scores = load_json(args.gt_scores_path) if args.gt_scores_path else None

    runner = TrainingFreeAblationRunner(base_config=config)
    report = runner.run(
        video_path=args.video_path,
        output_dir=args.output_dir,
        title=args.title,
        category=args.category,
        asr_segments=asr_segments,
        gt_scores=gt_scores,
    )
    print(f"Ablation report saved to: {report['report_path']}")


if __name__ == "__main__":
    main()