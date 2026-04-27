from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from src.data import DatasetVideoRecord, EvalVariantResult
from src.evaluation import EvaluationReporter, SplitEvaluationAggregator, VsumEvaluator
from src.io import JsonSaver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate archived inference results and print the best split-based results only."
    )
    parser.add_argument(
        "--archive_root",
        type=str,
        default=str(PROJECT_DIR / "archive" / "32s-deepseek"),
        help="Archive directory that contains inference_results/ and evaluation/.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/root/autodl-tmp/datasets",
        help="Dataset root containing H5 files and mapping JSON files.",
    )
    parser.add_argument(
        "--split_root",
        type=str,
        default="/root/autodl-tmp/datasets/splits",
        help="Split root containing official split JSON files.",
    )
    parser.add_argument(
        "--split_count",
        type=int,
        default=5,
        choices=[5, 50],
        help="Official split count.",
    )
    parser.add_argument(
        "--eval_smooth_window",
        type=int,
        default=5,
        help="Smoothing window used for normalized_smoothed.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["summe", "tvsum"],
        choices=["summe", "tvsum"],
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--exam_name_prefix",
        type=str,
        default=None,
        help="Optional exam name prefix. Final exam dirs become exam_<prefix>_<dataset>.",
    )
    parser.add_argument(
        "--video_name_dict_path",
        type=str,
        default=str(PROJECT_DIR.parent / "tfnet" / "data" / "video_name_dict.json"),
        help="Fallback mapping dictionary used when tvsum_mapping.json is unavailable.",
    )
    return parser


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def resolve_h5_path(dataset_root: Path, dataset_name: str) -> Path:
    file_name = (
        "eccv16_dataset_summe_google_pool5.h5"
        if dataset_name == "summe"
        else "eccv16_dataset_tvsum_google_pool5.h5"
    )
    return dataset_root / file_name


def load_mapping(dataset_root: Path, dataset_name: str) -> dict[str, str]:
    return load_json(dataset_root / f"{dataset_name}_mapping.json")


def sorted_mapping_keys(keys: list[str] | dict[str, str].keys) -> list[str]:
    def sort_key(value: str) -> tuple[int, str]:
        stem = Path(value).stem.lower()
        if stem.startswith("video_"):
            suffix = stem.split("_", 1)[1]
            if suffix.isdigit():
                return int(suffix), stem
        return 10**9, stem

    return sorted(keys, key=sort_key)


def find_video_key(mapping: dict[str, str], video_id: str) -> str:
    target = Path(video_id).stem.lower()
    for key, value in mapping.items():
        if key.lower() == target or Path(value).stem.lower() == target:
            return key
    raise KeyError(f"Failed to resolve dataset key for video_id={video_id}.")


def to_int_list(values: Any) -> list[int]:
    return np.asarray(values, dtype=np.int32).astype(int).tolist()


def to_nested_float_list(values: Any) -> list[list[float]]:
    return np.asarray(values, dtype=np.float32).astype(float).tolist()


def extract_user_scores(group: h5py.Group) -> list[list[float]] | list[float]:
    if "user_scores" in group:
        return to_nested_float_list(group["user_scores"][()])
    if "gtscore" in group:
        array = np.asarray(group["gtscore"][()], dtype=np.float32)
        return array.astype(float).tolist()
    return []


def build_mapping(
    dataset_root: Path,
    dataset_name: str,
    video_name_dict_path: Path | None,
) -> dict[str, str]:
    mapping_path = dataset_root / f"{dataset_name}_mapping.json"
    if mapping_path.is_file():
        return load_mapping(dataset_root, dataset_name)

    h5_path = resolve_h5_path(dataset_root, dataset_name)
    with h5py.File(h5_path, "r") as h5_file:
        dataset_keys = sorted_mapping_keys(list(h5_file.keys()))
        if dataset_name == "tvsum":
            if video_name_dict_path is None or not video_name_dict_path.is_file():
                raise FileNotFoundError(
                    "tvsum_mapping.json is missing and no usable video_name_dict.json fallback was provided."
                )
            raw_map = load_json(video_name_dict_path)
            video_id_to_name = {video_id: name for name, video_id in raw_map.items()}
            return {
                dataset_key: str(video_id_to_name.get(dataset_key, dataset_key))
                for dataset_key in dataset_keys
            }

        mapping: dict[str, str] = {}
        for dataset_key in dataset_keys:
            group = h5_file[dataset_key]
            video_name = group.get("video_name")
            value: Any = dataset_key
            if video_name is not None:
                raw_value = video_name[()]
                if isinstance(raw_value, bytes):
                    value = raw_value.decode("utf-8")
                elif isinstance(raw_value, np.ndarray) and raw_value.shape == ():
                    scalar = raw_value.item()
                    value = scalar.decode("utf-8") if isinstance(scalar, bytes) else scalar
                elif isinstance(raw_value, str):
                    value = raw_value
            mapping[dataset_key] = str(value)
        return mapping


def load_eval_record_by_key(
    dataset_root: Path,
    dataset_name: str,
    video_key: str,
    mapped_name: str,
) -> DatasetVideoRecord:
    h5_path = resolve_h5_path(dataset_root, dataset_name)

    with h5py.File(h5_path, "r") as h5_file:
        if video_key not in h5_file:
            raise KeyError(f"Failed to find video_key={video_key} in {h5_path}")
        group = h5_file[video_key]
        return DatasetVideoRecord(
            dataset_name=dataset_name,
            video_id=mapped_name,
            video_path="",
            n_frames=int(group["n_frames"][()]),
            picks=to_int_list(group["picks"][()]),
            change_points=to_int_list(group["change_points"][()]),
            n_frame_per_seg=to_int_list(group["n_frame_per_seg"][()]),
            user_summary=to_nested_float_list(group["user_summary"][()]),
            user_scores=extract_user_scores(group),
        )


def load_archived_scores(inference_dir: Path) -> tuple[list[int], list[float]]:
    frame_scores_path = inference_dir / "frame_scores.json"
    inference_result_path = inference_dir / "inference_result.json"

    if frame_scores_path.is_file():
        payload = load_json(frame_scores_path)
        picks = payload.get("picks", [])
        frame_scores = payload.get("frame_scores", [])
    elif inference_result_path.is_file():
        payload = load_json(inference_result_path)
        picks = payload.get("frame_score_picks", [])
        frame_scores = payload.get("frame_scores", [])
    else:
        raise FileNotFoundError(f"Missing frame_scores.json/inference_result.json under {inference_dir}")

    if not isinstance(picks, list) or not isinstance(frame_scores, list):
        raise ValueError(f"Invalid archived score payload under {inference_dir}")

    return [int(value) for value in picks], [float(value) for value in frame_scores]


def ensure_complete_archives(inference_root: Path, expected_video_ids: list[str], dataset_name: str) -> None:
    missing = [video_id for video_id in expected_video_ids if not (inference_root / video_id).is_dir()]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... total={len(missing)}"
        raise FileNotFoundError(
            f"Missing archived inference results for dataset={dataset_name}: {preview}{suffix}"
        )


def infer_splits_from_existing_overview(
    archive_root: Path,
    dataset_name: str,
    split_count: int,
) -> list[dict[str, Any]]:
    evaluation_root = archive_root / "evaluation"
    if not evaluation_root.is_dir():
        raise FileNotFoundError(
            f"Neither official split files nor existing split_overview.json were found for dataset={dataset_name}."
        )

    for split_overview_path in sorted(evaluation_root.rglob("split_overview.json")):
        payload = load_json(split_overview_path)
        if int(payload.get("split_count", 0)) != int(split_count):
            continue
        dataset_payload = payload.get("datasets", {}).get(dataset_name)
        if not isinstance(dataset_payload, dict):
            continue
        variants = dataset_payload.get("variants", {})
        if not isinstance(variants, dict) or not variants:
            continue
        first_variant = next(iter(variants.values()))
        split_results = first_variant.get("split_results", [])
        if split_results:
            return [
                {
                    "test_keys": list(split_result.get("test_keys", [])),
                    "train_keys": list(split_result.get("train_keys", [])),
                }
                for split_result in split_results
            ]

    raise FileNotFoundError(
        "Official split JSON files are missing, and no reusable split_overview.json was found "
        f"under {evaluation_root} for dataset={dataset_name}, split_count={split_count}."
    )


def ensure_split_inputs(
    archive_root: Path,
    dataset_root: Path,
    dataset_name: str,
    split_root: Path,
    split_count: int,
    video_name_dict_path: Path | None,
) -> Path:
    split_file = split_root / f"{dataset_name}_splits_{split_count}.json"
    mapping_file = split_root / f"{dataset_name}_mapping.json"
    if split_file.is_file() and mapping_file.is_file():
        return split_root

    generated_root = archive_root / "evaluation" / "_generated_split_inputs"
    json_saver = JsonSaver()
    mapping_payload = build_mapping(dataset_root, dataset_name, video_name_dict_path)
    if split_file.is_file():
        split_payload = load_json(split_file)
    else:
        split_payload = infer_splits_from_existing_overview(
            archive_root=archive_root,
            dataset_name=dataset_name,
            split_count=split_count,
        )

    json_saver.save(str(generated_root / f"{dataset_name}_mapping.json"), mapping_payload)
    json_saver.save(str(generated_root / f"{dataset_name}_splits_{split_count}.json"), split_payload)
    return generated_root


def evaluate_archived_dataset(
    dataset_name: str,
    archive_root: Path,
    dataset_root: Path,
    split_root: Path,
    split_count: int,
    smooth_window_size: int,
    exam_name_prefix: str,
    video_name_dict_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    inference_root = archive_root / "inference_results"
    evaluation_root = archive_root / "evaluation"
    evaluator = VsumEvaluator()
    reporter = EvaluationReporter(
        output_root=str(evaluation_root),
        exam_name=f"{exam_name_prefix}_{dataset_name}",
    )
    json_saver = JsonSaver()

    mapping = build_mapping(dataset_root, dataset_name, video_name_dict_path)
    ordered_pairs = [(video_key, mapping[video_key]) for video_key in sorted_mapping_keys(list(mapping.keys()))]
    ensure_complete_archives(inference_root, [video_id for _, video_id in ordered_pairs], dataset_name)

    for video_key, video_id in ordered_pairs:
        record = load_eval_record_by_key(dataset_root, dataset_name, video_key, video_id)
        archived_picks, original_scores = load_archived_scores(inference_root / video_id)

        if archived_picks != record.picks:
            raise ValueError(
                f"Pick alignment mismatch for dataset={dataset_name}, video_id={video_id}."
            )
        if len(original_scores) != len(record.picks):
            raise ValueError(
                f"Score length mismatch for dataset={dataset_name}, video_id={video_id}: "
                f"scores={len(original_scores)} picks={len(record.picks)}"
            )

        normalized_scores = reporter.normalize_scores(original_scores)
        smoothed_scores = reporter.smooth_scores(normalized_scores, window_size=smooth_window_size)
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
            original_frame_scores=original_scores,
            smooth_window_size=smooth_window_size,
        )
        json_saver.save(
            os.path.join(artifacts["video_dir"], f"eval_{Path(record.video_id).stem}.json"),
            {
                "exam_name": reporter.exam_name,
                "dataset_name": record.dataset_name,
                "video_id": record.video_id,
                "variants": [variant.to_dict() for variant in variants],
            },
        )

    split_aggregator = SplitEvaluationAggregator(
        split_root=str(
            ensure_split_inputs(
                archive_root=archive_root,
                dataset_root=dataset_root,
                dataset_name=dataset_name,
                split_root=split_root,
                split_count=split_count,
                video_name_dict_path=video_name_dict_path,
            )
        ),
        split_count=split_count,
    )
    split_summary = split_aggregator.aggregate_exam(
        exam_dir=reporter.exam_dir,
        datasets=[dataset_name],
    )
    overview_summary = load_json(Path(reporter.exam_dir) / "overview.json")
    return split_summary, overview_summary


def select_best_variant(dataset_summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    variants = dataset_summary.get("variants", {})
    if not variants:
        raise ValueError(f"No variants found in dataset summary: {dataset_summary.get('dataset_name', 'unknown')}")
    return max(
        variants.items(),
        key=lambda item: (
            float(item[1].get("mean_f1", 0.0)),
            float(item[1].get("mean_rho", 0.0)),
            float(item[1].get("mean_tau", 0.0)),
        ),
    )


def select_best_rank_variant(overview_summary: dict[str, Any], dataset_name: str) -> tuple[str, dict[str, Any]]:
    per_dataset_metrics = overview_summary.get("per_dataset_metrics", {})
    dataset_metrics = per_dataset_metrics.get(dataset_name, {})
    if not dataset_metrics:
        raise ValueError(f"No dataset metrics found in overview for dataset={dataset_name}")
    return max(
        dataset_metrics.items(),
        key=lambda item: (
            float(item[1].get("rho_mean", 0.0)),
            float(item[1].get("tau_mean", 0.0)),
            float(item[1].get("f1_mean", 0.0)),
        ),
    )


def format_best_result(
    dataset_name: str,
    split_dataset_summary: dict[str, Any],
    overview_summary: dict[str, Any],
) -> dict[str, Any]:
    _, split_variant_summary = select_best_variant(split_dataset_summary)
    _, rank_variant_summary = select_best_rank_variant(overview_summary, dataset_name)
    return {
        "mean_f1": float(split_variant_summary.get("mean_f1", 0.0)),
        "mean_f1_percent": round(float(split_variant_summary.get("mean_f1", 0.0)) * 100.0, 4),
        "mean_precision": float(split_variant_summary.get("mean_precision", 0.0)),
        "mean_precision_percent": round(float(split_variant_summary.get("mean_precision", 0.0)) * 100.0, 4),
        "mean_recall": float(split_variant_summary.get("mean_recall", 0.0)),
        "mean_recall_percent": round(float(split_variant_summary.get("mean_recall", 0.0)) * 100.0, 4),
        "mean_rho": float(rank_variant_summary.get("rho_mean", 0.0)),
        "mean_tau": float(rank_variant_summary.get("tau_mean", 0.0)),
    }


def main() -> None:
    args = build_parser().parse_args()

    archive_root = Path(args.archive_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    split_root = Path(args.split_root).resolve()
    exam_name_prefix = args.exam_name_prefix or datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name_dict_path = Path(args.video_name_dict_path).resolve() if args.video_name_dict_path else None

    results: dict[str, Any] = {}
    for dataset_name in args.datasets:
        split_summary, overview_summary = evaluate_archived_dataset(
            dataset_name=dataset_name,
            archive_root=archive_root,
            dataset_root=dataset_root,
            split_root=split_root,
            split_count=args.split_count,
            smooth_window_size=args.eval_smooth_window,
            exam_name_prefix=exam_name_prefix,
            video_name_dict_path=video_name_dict_path,
        )
        split_dataset_summary = split_summary.get("datasets", {}).get(dataset_name, {})
        results[dataset_name] = format_best_result(
            dataset_name=dataset_name,
            split_dataset_summary=split_dataset_summary,
            overview_summary=overview_summary,
        )

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()