"""Evaluate agent ablations for Chapter 4 using archived inference results.

This script reuses the archived 8s-deepseek inference outputs, recomputes scores
for each ablation variant, and produces the Chapter 4 agent-ablation table:
SumMe uses normalized_smoothed, TVSum uses normalized_raw.

Usage:
    python scripts/eval_ablation.py
    python scripts/eval_ablation.py --config 8s-deepseek --output archive/8s-deepseek/ablation_results/ablation_combined.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from run_offline_ablation import run_ablation


PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ARCHIVE_DIR = os.path.join(PROJECT_DIR, "archive")

ABLATION_ORDER = [
    "full",
    "w/o_planner",
    "w/o_experts",
    "w/o_story",
    "w/o_visual",
    "w/o_emotion",
    "w/o_information",
]

DISPLAY_NAMES = {
    "full": "Full model (Ours)",
    "w/o_planner": "w/o Planner Agent",
    "w/o_experts": "w/o Expert Agents",
    "w/o_story": "w/o Story Expert",
    "w/o_visual": "w/o Visual Expert",
    "w/o_emotion": "w/o Emotion Expert",
    "w/o_information": "w/o Information Expert",
}


def load_summary(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def build_paper_table(raw_summary: dict[str, Any], smooth_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    full_summe = float(smooth_summary["full"]["summe"])
    full_tvsum = float(raw_summary["full"]["tvsum"])

    for ablation_name in ABLATION_ORDER:
        summe_value = float(smooth_summary[ablation_name]["summe"])
        tvsum_value = float(raw_summary[ablation_name]["tvsum"])
        rows.append(
            {
                "ablation": ablation_name,
                "display_name": DISPLAY_NAMES[ablation_name],
                "summe": summe_value,
                "summe_delta": summe_value - full_summe,
                "tvsum": tvsum_value,
                "tvsum_delta": tvsum_value - full_tvsum,
            }
        )
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 96)
    print("Chapter 4 Agent Ablation")
    print("=" * 96)
    print(
        f"{'Model':<28}{'SumMe Max F1 (%)':<18}{'Δ':<10}"
        f"{'TVSum Avg F1 (%)':<18}{'Δ':<10}"
    )
    print("-" * 96)
    for row in rows:
        summe_delta = "---" if row["ablation"] == "full" else f"{row['summe_delta']:+.2f}"
        tvsum_delta = "---" if row["ablation"] == "full" else f"{row['tvsum_delta']:+.2f}"
        print(
            f"{row['display_name']:<28}{row['summe']:<18.2f}{summe_delta:<10}"
            f"{row['tvsum']:<18.2f}{tvsum_delta:<10}"
        )


def save_results(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Chapter 4 agent ablation from archived inference results")
    parser.add_argument("--config", default="8s-deepseek", help="Config name under archive/")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Defaults to archive/<config>/ablation_results/ablation_combined.json",
    )
    args = parser.parse_args()

    raw_summary = run_ablation(args.config, variant="normalized_raw")
    smooth_summary = run_ablation(args.config, variant="normalized_smoothed")

    if raw_summary is None or smooth_summary is None:
        raise RuntimeError("Ablation evaluation failed. Please check archive paths and H5 datasets.")

    paper_table = build_paper_table(raw_summary, smooth_summary)
    print_table(paper_table)

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(ARCHIVE_DIR, args.config, "ablation_results", "ablation_combined.json")

    raw_json_path = os.path.join(ARCHIVE_DIR, args.config, "ablation_results", "ablation_normalized_raw.json")
    smooth_json_path = os.path.join(ARCHIVE_DIR, args.config, "ablation_results", "ablation_normalized_smoothed.json")

    payload = {
        "config": args.config,
        "paper_protocol": {
            "summe_variant": "normalized_smoothed",
            "tvsum_variant": "normalized_raw",
        },
        "paper_table": paper_table,
        "raw_summary": raw_summary,
        "smoothed_summary": smooth_summary,
        "source_files": {
            "raw": raw_json_path,
            "smoothed": smooth_json_path,
        },
    }
    save_results(output_path, payload)
    print(f"\nSaved combined ablation summary to: {output_path}")


if __name__ == "__main__":
    main()