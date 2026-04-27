"""
Offline ablation study for the multi-agent video summarization framework.

Recomputes segment-level final scores by removing individual agents,
then evaluates against GT using the standard knapsack + F1 pipeline.

Usage:
    python scripts/run_offline_ablation.py --config 8s-deepseek
    python scripts/run_offline_ablation.py --config 8s-deepseek --dataset summe
"""
import argparse
import json
import math
import os
import sys

import h5py
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────
ARCHIVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "archive")
H5_DIR = os.path.expanduser("~/Resources/datasets")
H5_FILES = {
    "summe": os.path.join(H5_DIR, "eccv16_dataset_summe_google_pool5.h5"),
    "tvsum": os.path.join(H5_DIR, "eccv16_dataset_tvsum_google_pool5.h5"),
}
TVSUM_NAME_DICT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "tfnet", "data", "video_name_dict.json"
)

EXPERT_NAMES = ["story_agent", "visual_agent", "emotion_agent", "information_agent"]


# ── knapsack ─────────────────────────────────────────────────────────────
def knapsack_dp(values, weights, capacity):
    n = len(weights)
    if capacity <= 0 or n == 0:
        return []
    int_values = [int(v * 1000) for v in values]
    # Use 1D DP with traceback
    dp = [0] * (capacity + 1)
    # Store decisions for traceback
    decisions = [[False] * (capacity + 1) for _ in range(n)]
    for i in range(n):
        w = weights[i]
        v = int_values[i]
        if w > capacity:
            continue
        for c in range(capacity, w - 1, -1):
            new_val = dp[c - w] + v
            if new_val > dp[c]:
                dp[c] = new_val
                decisions[i][c] = True
    selected = []
    c = capacity
    for i in range(n - 1, -1, -1):
        if decisions[i][c]:
            selected.append(i)
            c -= weights[i]
    return selected[::-1]


def compute_f1(machine_summary, user_summary, reduction):
    machine = np.asarray(machine_summary, dtype=np.float32)
    humans = np.asarray(user_summary, dtype=np.float32)
    machine[machine > 0] = 1
    humans[humans > 0] = 1
    n_users, n_frames = humans.shape
    if machine.size < n_frames:
        machine = np.pad(machine, (0, n_frames - machine.size))
    else:
        machine = machine[:n_frames]

    f1s = []
    for u in range(n_users):
        overlap = np.sum(machine * humans[u])
        prec = overlap / (np.sum(machine) + 1e-8)
        rec = overlap / (np.sum(humans[u]) + 1e-8)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)

    if reduction == "max":
        return float(max(f1s))
    return float(np.mean(f1s))


def build_summary_and_f1(frame_scores, picks, change_points, n_frames,
                         n_frame_per_seg, user_summary, reduction):
    """From frame-level scores to F1."""
    pred = np.asarray(frame_scores, dtype=np.float32)
    picks_arr = np.asarray(picks, dtype=np.int32)
    cps = np.asarray(change_points, dtype=np.int32)
    nfps = list(np.asarray(n_frame_per_seg, dtype=np.int32))

    sampled = np.array(picks_arr)
    if sampled[-1] != n_frames:
        sampled = np.append(sampled, n_frames)

    # frame-level scores from picks
    frame_level = np.zeros(n_frames, dtype=np.float32)
    for i in range(len(pred)):
        frame_level[sampled[i]:sampled[i + 1]] = pred[i]

    # segment-level mean
    seg_scores = [float(frame_level[s:e + 1].mean()) for s, e in cps]
    capacity = int(math.floor(n_frames * 0.15))

    selected = knapsack_dp(seg_scores, nfps, capacity)
    summary = np.concatenate([
        np.ones(nfps[i], dtype=np.float32) if i in selected
        else np.zeros(nfps[i], dtype=np.float32)
        for i in range(len(nfps))
    ])

    return compute_f1(summary, user_summary, reduction)


# ── ablation variants ────────────────────────────────────────────────────
def recompute_final_scores(segment_scores, planner_plan, ablation_type):
    """
    Recompute final_score for each segment under a given ablation.

    ablation_type:
        "full"           - full model
        "w/o_planner"    - experts only (no planner score)
        "w/o_experts"    - planner only (no expert scores)
        "w/o_story"      - remove story expert
        "w/o_visual"     - remove visual expert
        "w/o_emotion"    - remove emotion expert
        "w/o_information"- remove information expert
        "w/o_memory"     - same as full (memory can't be removed offline)
    """
    weights = planner_plan.get("expert_weights", {})
    segments = segment_scores["segments"]
    new_scores = []

    for seg in segments:
        planner_score = float(seg["planner_score"])
        expert_results = seg["expert_results"]

        if ablation_type == "full" or ablation_type == "w/o_memory":
            new_scores.append(float(seg["final_score"]))
            continue

        if ablation_type == "w/o_planner":
            # Only expert scores, weighted
            total_w = sum(weights.get(e, 0.25) for e in EXPERT_NAMES)
            score = 0.0
            for e in EXPERT_NAMES:
                w = weights.get(e, 0.25) / max(total_w, 1e-8)
                score += w * float(expert_results[e]["score"])
            new_scores.append(score)
            continue

        if ablation_type == "w/o_experts":
            new_scores.append(planner_score)
            continue

        # Remove a specific expert
        removed = ablation_type.replace("w/o_", "") + "_agent"
        active_experts = [e for e in EXPERT_NAMES if e != removed]
        active_w = {e: weights.get(e, 0.25) for e in active_experts}
        total_w = sum(active_w.values())
        if total_w > 0:
            active_w = {e: w / total_w for e, w in active_w.items()}

        expert_sum = sum(active_w[e] * float(expert_results[e]["score"])
                         for e in active_experts)
        new_scores.append(planner_score + expert_sum)

    return new_scores


def map_segment_to_frame_scores(segment_final_scores, inference_result):
    """Map segment scores back to frame-level scores using picks."""
    picks = inference_result["frame_score_picks"]
    segments = inference_result["segment_scores"]
    n_picks = len(picks)

    frame_scores = np.zeros(n_picks, dtype=np.float32)
    for seg_idx, seg in enumerate(segments):
        start = seg["start_frame"]
        end = seg["end_frame"]
        score = segment_final_scores[seg_idx]
        for i, p in enumerate(picks):
            if start <= p < end:
                frame_scores[i] = score

    return frame_scores.tolist()


def normalize_scores(scores):
    arr = np.asarray(scores, dtype=np.float32)
    lo, hi = arr.min(), arr.max()
    if hi - lo <= 1e-8:
        return [0.0] * len(scores)
    return ((arr - lo) / (hi - lo)).tolist()


def smooth_scores(scores, window=5):
    arr = np.asarray(scores, dtype=np.float32)
    if len(arr) <= window:
        return arr.tolist()
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid").tolist()


# ── video name mapping ───────────────────────────────────────────────────
def build_video_mappings():
    """Build mappings: inference_dir_name -> (dataset, h5_key)."""
    mapping = {}

    # SumMe
    h5 = h5py.File(H5_FILES["summe"], "r")
    for key in h5.keys():
        vname = h5[key]["video_name"][()].decode()
        mapping[vname] = ("summe", key)
        # Also try space-replaced version
        mapping[vname.replace(" ", "_")] = ("summe", key)
    h5.close()

    # TVSum
    if os.path.exists(TVSUM_NAME_DICT):
        with open(TVSUM_NAME_DICT) as f:
            tvsum_dict = json.load(f)
        for yt_id, vkey in tvsum_dict.items():
            mapping[yt_id] = ("tvsum", vkey)

    return mapping


# ── main ─────────────────────────────────────────────────────────────────
def run_ablation(config_name, target_dataset=None, variant="normalized_raw"):
    inference_dir = os.path.join(ARCHIVE_DIR, config_name, "inference")
    if not os.path.isdir(inference_dir):
        inference_dir = os.path.join(ARCHIVE_DIR, config_name, "inference_results")

    if not os.path.isdir(inference_dir):
        print(f"ERROR: Inference dir not found: {inference_dir}")
        return

    video_mapping = build_video_mappings()

    ablation_types = [
        "full",
        "w/o_planner",
        "w/o_experts",
        "w/o_story",
        "w/o_visual",
        "w/o_emotion",
        "w/o_information",
    ]

    h5_handles = {
        ds: h5py.File(path, "r") for ds, path in H5_FILES.items()
    }

    # Extract split definitions from existing split_overview
    splits_def = extract_splits(config_name)

    # Per-video F1 cache: {(abl, video_id): f1}
    video_f1_cache = {}

    video_dirs = sorted(os.listdir(inference_dir))
    for vdir in video_dirs:
        vpath = os.path.join(inference_dir, vdir)
        if not os.path.isdir(vpath):
            continue

        seg_path = os.path.join(vpath, "segment_scores.json")
        plan_path = os.path.join(vpath, "planner_plan.json")
        inf_path = os.path.join(vpath, "inference_result.json")
        fs_path = os.path.join(vpath, "frame_scores.json")

        if not all(os.path.exists(p) for p in [seg_path, plan_path, inf_path, fs_path]):
            continue

        with open(seg_path) as f:
            seg_data = json.load(f)
        with open(plan_path) as f:
            plan_data = json.load(f)
        with open(inf_path) as f:
            inf_data = json.load(f)
        with open(fs_path) as f:
            fs_data = json.load(f)

        # Determine dataset
        dataset, h5_key = None, None
        for name_variant in [vdir, vdir.replace("_", " "), vdir.replace(" ", "_")]:
            if name_variant in video_mapping:
                dataset, h5_key = video_mapping[name_variant]
                break
        if dataset is None or (target_dataset and dataset != target_dataset):
            continue

        h5 = h5_handles[dataset]
        video_data = h5[h5_key]
        picks = np.asarray(video_data["picks"])
        cps = np.asarray(video_data["change_points"])
        nfps = np.asarray(video_data["n_frame_per_seg"]).tolist()
        n_frames = int(np.asarray(video_data["n_frames"]))
        user_summary = np.asarray(video_data["user_summary"], dtype=np.float32)
        reduction = "max" if dataset == "summe" else "avg"

        for abl in ablation_types:
            new_seg_scores = recompute_final_scores(seg_data, plan_data, abl)
            frame_scores = map_segment_to_frame_scores(new_seg_scores, inf_data)

            frame_scores = normalize_scores(frame_scores)
            if variant == "normalized_smoothed":
                frame_scores = smooth_scores(frame_scores)

            our_picks = fs_data["picks"]
            our_pick_map = {p: i for i, p in enumerate(our_picks)}
            aligned = np.array([
                frame_scores[our_pick_map[int(p)]] if int(p) in our_pick_map else 0.0
                for p in picks
            ], dtype=np.float32)

            f1 = build_summary_and_f1(
                aligned, picks, cps, n_frames, nfps, user_summary, reduction
            )
            # Store with normalized video_id
            norm_vid = vdir.replace(" ", "_")
            video_f1_cache[(abl, dataset, norm_vid)] = f1

    for h5 in h5_handles.values():
        h5.close()

    # Compute split-based 5-fold CV mean (matching paper protocol)
    print(f"\n{'='*80}")
    print(f"Ablation Results (5-fold CV): {config_name} ({variant})")
    print(f"{'='*80}")

    summary = {}
    for abl in ablation_types:
        for ds in ["summe", "tvsum"]:
            if ds not in splits_def:
                continue
            split_f1s = []
            for split_videos in splits_def[ds]:
                split_video_f1s = []
                for vid in split_videos:
                    norm_vid = vid.replace(" ", "_")
                    f1 = video_f1_cache.get((abl, ds, norm_vid))
                    if f1 is not None:
                        split_video_f1s.append(f1)
                if split_video_f1s:
                    split_f1s.append(np.mean(split_video_f1s))
            if split_f1s:
                summary.setdefault(abl, {})[ds] = np.mean(split_f1s) * 100

        # Also compute per-video mean
        for ds in ["summe", "tvsum"]:
            all_f1 = [v for (a, d, _), v in video_f1_cache.items() if a == abl and d == ds]
            if all_f1:
                summary.setdefault(abl, {})[f"{ds}_pervid"] = np.mean(all_f1) * 100

    # Print table
    print(f"\n{'Ablation':<25s} {'SumMe Max F1 (%)':<20s} {'TVSum Avg F1 (%)':<20s}")
    print("-" * 65)
    for abl in ablation_types:
        summe_f1 = summary.get(abl, {}).get("summe", 0)
        tvsum_f1 = summary.get(abl, {}).get("tvsum", 0)
        s_str = f"{summe_f1:.2f}" if summe_f1 > 0 else "---"
        t_str = f"{tvsum_f1:.2f}" if tvsum_f1 > 0 else "---"
        print(f"{abl:<25s} {s_str:<20s} {t_str:<20s}")

    full_summe = summary.get("full", {}).get("summe", 0)
    full_tvsum = summary.get("full", {}).get("tvsum", 0)
    print(f"\n{'Ablation':<25s} {'ΔSumMe':<12s} {'ΔTVSum':<12s}")
    print("-" * 49)
    for abl in ablation_types:
        if abl == "full":
            continue
        ds = summary.get(abl, {}).get("summe", 0) - full_summe
        dt = summary.get(abl, {}).get("tvsum", 0) - full_tvsum
        print(f"{abl:<25s} {ds:+.2f}{'':>6s} {dt:+.2f}")

    out_dir = os.path.join(ARCHIVE_DIR, config_name, "ablation_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ablation_{variant}.json")
    with open(out_path, "w") as f:
        json.dump({"config": config_name, "variant": variant,
                   "summary": summary,
                   "per_video": {
                       f"{abl}_{ds}_{vid}": f1
                       for (abl, ds, vid), f1 in video_f1_cache.items()
                   }}, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")

    return summary


def extract_splits(config_name):
    """Extract split definitions from existing split_overview.json files."""
    splits_def = {}
    base = os.path.join(ARCHIVE_DIR, config_name)

    # Search for split_overview.json files
    for root, dirs, files in os.walk(base):
        if "split_overview.json" in files:
            with open(os.path.join(root, "split_overview.json")) as f:
                data = json.load(f)
            for ds_name, ds_data in data.get("datasets", {}).items():
                if ds_name in splits_def:
                    continue
                # Extract test video IDs per split
                for variant_name, variant_data in ds_data.get("variants", {}).items():
                    split_videos = []
                    for sr in variant_data.get("split_results", []):
                        split_videos.append(sr["video_ids"])
                    if split_videos:
                        splits_def[ds_name] = split_videos
                    break  # Only need one variant for split defs

    return splits_def


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="8s-deepseek", help="Config name in archive/")
    parser.add_argument("--dataset", default=None, help="summe or tvsum (default: both)")
    parser.add_argument("--variant", default="normalized_raw",
                        choices=["normalized_raw", "normalized_smoothed"])
    args = parser.parse_args()
    run_ablation(args.config, args.dataset, args.variant)
