"""YAML configuration loading, merging and runtime normalization utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

import yaml


class ConfigNode(dict):
    """Dictionary wrapper that provides attribute-style recursive access."""

    def __init__(self, initial: Mapping[str, Any] | None = None):
        super().__init__()
        if initial:
            for key, value in initial.items():
                self[key] = value

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - mirrors dict semantics
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, self._wrap(value))

    def copy(self) -> "ConfigNode":
        return ConfigNode(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, ConfigNode):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, ConfigNode) else item for item in value]
            else:
                result[key] = value
        return result

    def _wrap(self, value: Any) -> Any:
        if isinstance(value, ConfigNode):
            return value
        if isinstance(value, Mapping):
            return ConfigNode(value)
        if isinstance(value, list):
            return [self._wrap(item) for item in value]
        return value


def load_config(config_path: str | Path) -> ConfigNode:
    """Load a YAML config with optional recursive `defaults` includes."""
    resolved_path = Path(config_path).expanduser().resolve()
    if not resolved_path.exists():
        return build_runtime_config({})

    loaded = _load_yaml_with_defaults(resolved_path)
    return build_runtime_config(loaded)


def build_runtime_config(config_data: Mapping[str, Any] | None) -> ConfigNode:
    """Normalize a raw config dictionary into the canonical runtime structure."""
    base = deepcopy(dict(config_data or {}))
    normalized = _default_skeleton()
    normalized = deep_merge(normalized, base)
    normalized = _apply_legacy_aliases(normalized)
    return ConfigNode(normalized)


def set_nested_value(config: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested config field using a dotted key path."""
    current: MutableMapping[str, Any] = config
    path_parts = dotted_key.split(".")
    for part in path_parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, MutableMapping):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[path_parts[-1]] = value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two nested mappings."""
    merged = deepcopy(dict(base))
    for key, value in dict(override).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml_with_defaults(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj) or {}

    defaults = payload.pop("defaults", []) or []
    merged: dict[str, Any] = {}
    for entry in defaults:
        default_path = (config_path.parent / str(entry)).resolve()
        merged = deep_merge(merged, _load_yaml_with_defaults(default_path))
    return deep_merge(merged, payload)


def _default_skeleton() -> dict[str, Any]:
    return {
        "project": {"name": "VideoSummarizationAgent", "schema_version": "2.0", "seed": 42},
        "paths": {
            "output_root": "outputs",
            "summary_output_dir": "outputs/inference_results",
            "evaluation_output_dir": "outputs/evaluation",
            "ablation_output_dir": "outputs/ablation",
            "checkpoint_dir": "checkpoints",
        },
        "dataset": {
            "name": "custom_video_folder",
            "data_root": "data/raw",
            "video_path": "data/raw/demo.mp4",
            "metadata_file": None,
            "metadata_path": "data/metadata",
            "split": "val",
            "category_field_optional": True,
            "asr_segments_optional": True,
        },
        "video": {
            "num_frames": 32,
            "resolution": 224,
            "frame_sample_rate": 1,
            "global_sample_rate": 2,
            "expected_time_units": ["frame", "second"],
        },
        "model": {
            "name": "resnet18",
            "feature_extractor": "resnet18",
            "embed_dim": 512,
            "pretrained": False,
            "weights_path": None,
        },
        "agent": {
            "backend": "rule_based",
            "hidden_dim": 256,
            "action_space": 2,
            "num_layers": 2,
            "roles": ["planner", "mainline", "novelty", "event", "temporal", "domain", "critic"],
            "history_length": 0,
            "llm_model_name": None,
            "temperature": 0.0,
            "max_tokens": 0,
            "prompts": {},
        },
        "memory": {"topk": 5, "max_items_per_slot": 50, "similarity_prune_threshold": 0.9},
        "summarization": {
            "segment_length_sec": 30,
            "budget_ratio": 0.15,
            "window": {"length_sec": 8, "overlap_sec": 2, "sample_rate": 1},
            "selection": {"min_label": "low", "allow_partial_segment": True},
            "aggregation": {
                "score_weights": {
                    "mainline": {"强": 3.0, "中": 1.5, "弱": 0.0},
                    "novelty": {"高": 2.0, "中": 1.0, "低": 0.0},
                    "event": {"高": 2.0, "中": 1.0, "低": 0.0},
                    "temporal": {"高": 1.0, "中": 0.5, "低": 0.0},
                    "domain": {"高": 1.0, "中": 0.5, "低": 0.0},
                    "critic": {"高": 1.5, "中": 0.5, "低": -0.5},
                },
                "decision_thresholds": {"suggest_keep": 4.5, "optional": 2.0},
            },
        },
        "experiment": {
            "mode": "inference",
            "epochs": 10,
            "batch_size": 2,
            "lr": 0.0001,
            "log_interval": 10,
            "num_workers": 0,
            "device": "auto",
            "seed": 42,
        },
        "evaluation": {
            "num_samples": 5,
            "output_dir": "outputs/evaluation",
            "metrics": ["precision", "recall", "fscore", "coverage", "diversity", "spearman", "kendall"],
            "ablation_variants": {
                "full_system": {},
                "no_planner": {"ablation": {"disable_planner": True}},
                "no_critic": {"ablation": {"disable_critic": True}},
                "no_domain": {"ablation": {"disable_domain": True}},
                "no_memory": {"ablation": {"disable_memory": True}},
                "compact_memory": {"memory": {"max_items_per_slot": 8, "similarity_prune_threshold": 0.7}},
            },
        },
        "logging": {"level": "INFO", "wandb": {"enabled": False, "project": "video-agent-project"}},
        "ablation": {"disable_planner": False, "disable_critic": False, "disable_domain": False, "disable_memory": False},
        "llm": {"backend": "rule_based", "model_name": None, "temperature": 0.0, "max_tokens": 0},
    }


def _apply_legacy_aliases(config: dict[str, Any]) -> dict[str, Any]:
    if "data_root" in config:
        config["dataset"]["data_root"] = config["data_root"]
    if "video_path" in config:
        config["dataset"]["video_path"] = config["video_path"]
    if "metadata_file" in config:
        config["dataset"]["metadata_file"] = config["metadata_file"]
    if "num_samples" in config:
        config["evaluation"]["num_samples"] = config["num_samples"]
    if "epochs" in config:
        config["experiment"]["epochs"] = config["epochs"]
    if "model_config" in config and isinstance(config["model_config"], Mapping):
        config["model"] = deep_merge(config["model"], config["model_config"])
    if "agent_config" in config and isinstance(config["agent_config"], Mapping):
        config["agent"] = deep_merge(config["agent"], config["agent_config"])
    if "training" in config and isinstance(config["training"], Mapping):
        config["experiment"] = deep_merge(config["experiment"], config["training"])
    if "segment" in config and isinstance(config["segment"], Mapping):
        segment_length = config["segment"].get("coarse_segment_sec")
        if segment_length is not None:
            config["summarization"]["segment_length_sec"] = segment_length
    if "window" in config and isinstance(config["window"], Mapping):
        config["summarization"]["window"]["length_sec"] = config["window"].get("win_len_sec", config["summarization"]["window"].get("length_sec"))
        config["summarization"]["window"]["overlap_sec"] = config["window"].get("overlap_sec", config["summarization"]["window"].get("overlap_sec"))
        config["summarization"]["window"]["sample_rate"] = config["window"].get("sample_rate", config["summarization"]["window"].get("sample_rate"))
    if "summary" in config and isinstance(config["summary"], Mapping):
        budget_ratio = config["summary"].get("budget_ratio")
        if budget_ratio is not None:
            config["summarization"]["budget_ratio"] = budget_ratio
    if "optimization" in config and isinstance(config["optimization"], Mapping):
        if isinstance(config["optimization"].get("selection"), Mapping):
            config["summarization"]["selection"] = deep_merge(config["summarization"]["selection"], config["optimization"]["selection"])
        if isinstance(config["optimization"].get("aggregation"), Mapping):
            config["summarization"]["aggregation"] = deep_merge(config["summarization"]["aggregation"], config["optimization"]["aggregation"])
    if "eval_output_dir" in config:
        config["evaluation"]["output_dir"] = config["eval_output_dir"]
    if isinstance(config.get("logging"), Mapping) and "wandb_project" in config["logging"]:
        config["logging"]["wandb"]["project"] = config["logging"]["wandb_project"]

    config["data_root"] = config["dataset"]["data_root"]
    config["video_path"] = config["dataset"]["video_path"]
    config["metadata_file"] = config["dataset"].get("metadata_file")
    config["num_samples"] = config["evaluation"]["num_samples"]
    config["epochs"] = config["experiment"]["epochs"]
    config["training"] = deepcopy(config["experiment"])
    config["model_config"] = deepcopy(config["model"])
    config["agent_config"] = deepcopy(config["agent"])
    config["segment"] = {"coarse_segment_sec": config["summarization"]["segment_length_sec"]}
    config["window"] = {
        "win_len_sec": config["summarization"]["window"]["length_sec"],
        "overlap_sec": config["summarization"]["window"]["overlap_sec"],
        "sample_rate": config["summarization"]["window"]["sample_rate"],
    }
    config["summary"] = {"budget_ratio": config["summarization"]["budget_ratio"]}
    config["optimization"] = {
        "selection": deepcopy(config["summarization"]["selection"]),
        "aggregation": deepcopy(config["summarization"]["aggregation"]),
    }
    config["eval_output_dir"] = config["evaluation"]["output_dir"]
    return config