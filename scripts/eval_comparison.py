"""Summarize Chapter 4 comparison/configuration results from archived split reports.

This script reads existing split_overview.json files under archive/ and builds:
1. The configuration ablation table used in Chapter 4.
2. The best "Ours" results used in the benchmark comparison table.

Usage:
    python scripts/eval_comparison.py
    python scripts/eval_comparison.py --configs 8s-deepseek 16s-deepseek
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ARCHIVE_DIR = os.path.join(PROJECT_DIR, "archive")


def find_split_overview_files(config_name: str) -> list[str]:
    config_dir = os.path.join(ARCHIVE_DIR, config_name)
    matches: list[str] = []
    if not os.path.isdir(config_dir):
        return matches

    for root, _, files in os.walk(config_dir):
        if "split_overview.json" in files:
            matches.append(os.path.join(root, "split_overview.json"))
    return sorted(matches)


def extract_config_rows(config_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_path in find_split_overview_files(config_name):
        with open(split_path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)

        datasets = payload.get("datasets", {})
        for dataset_name, dataset_payload in datasets.items():
            variants = dataset_payload.get("variants", {})
            for variant_name, variant_payload in variants.items():
                rows.append(
                    {
                        "config": config_name,
                        "dataset": dataset_name.lower(),
                        "variant": variant_name,
                        "mean_f1": float(variant_payload.get("mean_f1", 0.0)) * 100.0,
                        "num_splits": int(variant_payload.get("num_splits", 0)),
                        "source": split_path,
                    }
                )
    return rows


def parse_segment_count(config_name: str) -> int | None:
    token = config_name.split("-", 1)[0]
    if token.endswith("s") and token[:-1].isdigit():
        return int(token[:-1])
    return None


def build_config_matrix(rows: list[dict[str, Any]], configs: list[str]) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["config"] in configs]
    grouped: dict[tuple[int, str], dict[str, Any]] = {}

    for row in selected:
        segment_count = parse_segment_count(row["config"])
        if segment_count is None:
            continue
        variant_label = "Raw" if row["variant"] == "normalized_raw" else "Smoothed"
        key = (segment_count, variant_label)
        bucket = grouped.setdefault(
            key,
            {
                "segment_count": segment_count,
                "postprocess": variant_label,
                "summe": None,
                "tvsum": None,
                "sources": {},
            },
        )
        bucket[row["dataset"]] = row["mean_f1"]
        bucket["sources"][row["dataset"]] = row["source"]

    return sorted(grouped.values(), key=lambda item: (item["segment_count"], item["postprocess"] != "Raw"))


def pick_best_results(rows: list[dict[str, Any]], configs: list[str]) -> dict[str, dict[str, Any]]:
    candidates = [row for row in rows if row["config"] in configs]
    best: dict[str, dict[str, Any]] = {}
    for dataset_name in ("summe", "tvsum"):
        dataset_rows = [row for row in candidates if row["dataset"] == dataset_name]
        if not dataset_rows:
            continue
        best[dataset_name] = max(dataset_rows, key=lambda item: item["mean_f1"])
    return best


def build_paper_best(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    desired = {
        ("summe", "8s-deepseek", "normalized_smoothed"),
        ("tvsum", "16s-deepseek", "normalized_raw"),
    }
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = (row["dataset"], row["config"], row["variant"])
        if key in desired:
            result[row["dataset"]] = row
    return result


def format_score(value: float | None) -> str:
    return "---" if value is None else f"{value:.2f}"


def print_config_table(matrix: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 76)
    print("Chapter 4 Configuration Ablation")
    print("=" * 76)
    print(f"{'N':<8}{'Postprocess':<16}{'SumMe Max F1 (%)':<24}{'TVSum Avg F1 (%)':<24}")
    print("-" * 76)
    for row in matrix:
        print(
            f"{row['segment_count']:<8}{row['postprocess']:<16}"
            f"{format_score(row['summe']):<24}{format_score(row['tvsum']):<24}"
        )


def print_best_table(title: str, best: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)
    print(f"{'Dataset':<12}{'F1 (%)':<12}{'Config':<20}{'Variant':<24}")
    print("-" * 76)
    for dataset_name in ("summe", "tvsum"):
        row = best.get(dataset_name)
        if row is None:
            continue
        print(
            f"{dataset_name:<12}{row['mean_f1']:<12.2f}"
            f"{row['config']:<20}{row['variant']:<24}"
        )


def save_results(output_path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize archived comparison results for Chapter 4")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["8s-deepseek", "16s-deepseek"],
        help="Configs to use for Chapter 4 comparison tables",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(ARCHIVE_DIR, "comparison_results.json"),
        help="Path to save the aggregated JSON summary",
    )
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    for config_name in args.configs:
        all_rows.extend(extract_config_rows(config_name))

    if not all_rows:
        raise FileNotFoundError("No split_overview.json files were found for the requested configs.")

    matrix = build_config_matrix(all_rows, args.configs)
    paper_best = build_paper_best(all_rows)
    overall_best = pick_best_results(all_rows, args.configs)

    print_config_table(matrix)
    print_best_table("Chapter 4 Ours Results", paper_best)
    print_best_table("Best Within Selected Configs", overall_best)

    save_results(
        args.output,
        {
            "configs": args.configs,
            "configuration_ablation": matrix,
            "chapter4_ours": paper_best,
            "best_within_selected_configs": overall_best,
            "raw_rows": all_rows,
        },
    )
    print(f"\nSaved comparison summary to: {args.output}")


if __name__ == "__main__":
    main()