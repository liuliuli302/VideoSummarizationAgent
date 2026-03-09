from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import torch
from PIL import Image
from tqdm.auto import tqdm
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)


DEFAULT_SCENE_PROMPT = (
'''
You are given several frames sampled from a short video.

Describe the overall content of the video based on these frames.

Requirements:
- Focus on the main actions and events.
- Mention important objects, people, and scene context.
- Do not speculate beyond the visual content.
- Produce one coherent paragraph summarizing the video.

Keep the description concise (80-120 words)."
'''
)


class QwenSceneCaptionConfig:
    prompt = DEFAULT_SCENE_PROMPT
    num_frames = 10
    max_new_tokens = 120
    scene_threshold = 27.0


class LlavaSceneCaptionConfig:
    prompt = DEFAULT_SCENE_PROMPT
    num_frames = 10
    max_new_tokens = 120
    scene_threshold = 27.0


def detect_scenes(video_path, threshold=27.0):

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames <= 0:
        return []

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    scenes = [
        (start.get_frames(), max(start.get_frames(), end.get_frames() - 1))
        for start, end in scene_list
    ]

    if not scenes:
        return [(0, max(total_frames - 1, 0))]

    return scenes


def _sample_frame_indices(start_frame, end_frame, num_frames):

    if num_frames <= 0 or end_frame < start_frame:
        return []

    if start_frame == end_frame:
        return [int(start_frame)]

    total = end_frame - start_frame + 1
    count = min(num_frames, total)

    if count == 1:
        return [int(start_frame)]

    return [
        int(round(start_frame + i * (total - 1) / (count - 1)))
        for i in range(count)
    ]


def sample_scene_frames(video_path, start_frame, end_frame, num_frames=8):

    frame_indices = _sample_frame_indices(start_frame, end_frame, num_frames)

    if not frame_indices:
        return []

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    images = []

    for frame_idx in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()

        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        images.append(img)

    cap.release()
    return images


class QwenSceneCaptioner:

    @staticmethod
    def _postprocess_caption(text, prompt):

        text = text.strip()

        if prompt:
            prompt = prompt.strip()
            if text.lower().startswith(prompt.lower()):
                text = text[len(prompt):].strip(" \n:\t")

        if "assistant" in text.lower() and ":" in text:
            text = text.split(":", 1)[-1].strip()

        return text

    def __init__(
        self,
        model_name="Qwen/Qwen3-VL-8B-Instruct",
        device="auto",
        config=QwenSceneCaptionConfig,
    ):

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.config = config

        print("Loading Qwen Scene Caption:", model_name)

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.dtype
        ).to(device)

        self.model.eval()

    def caption_frames(self, images, prompt, max_new_tokens):

        prompt = (prompt or "").strip()

        if not prompt:
            prompt = self.config.prompt

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in images] + [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True
        )

        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)

        with torch.inference_mode():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )

        generated_ids = ids[:, inputs["input_ids"].shape[1]:]
        captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return self._postprocess_caption(captions[0], prompt)

    def generate_caption(self, frames, prompt):

        if not frames:
            return ""

        return self.caption_frames(
            list(frames),
            prompt,
            self.config.max_new_tokens
        )

    def caption_video_scenes(self, video_path, prompt=None, num_frames=None, max_new_tokens=None, scene_threshold=None):

        if num_frames is None:
            num_frames = self.config.num_frames

        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens

        if scene_threshold is None:
            scene_threshold = self.config.scene_threshold

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames <= 0:
            return {
                "video_path": video_path,
                "fps": fps,
                "total_frames": total_frames,
                "scene_threshold": scene_threshold,
                "num_scenes": 0,
                "scenes": []
            }

        scene_ranges = detect_scenes(video_path, threshold=scene_threshold)
        results = []

        progress = tqdm(
            scene_ranges,
            desc=f"Captioning scenes ({Path(video_path).stem})",
            unit="scene",
            leave=False
        )

        for i, (start_frame, end_frame) in enumerate(progress):

            sampled_frame_indices = _sample_frame_indices(
                start_frame,
                end_frame,
                num_frames
            )

            images = sample_scene_frames(
                video_path,
                start_frame,
                end_frame,
                num_frames
            )

            if not images:
                continue

            caption = self.caption_frames(
                images,
                prompt,
                max_new_tokens
            )

            results.append({
                "scene_index": i,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start_sec": round(start_frame / fps, 6) if fps else None,
                "end_sec": round(end_frame / fps, 6) if fps else None,
                "duration_sec": round((end_frame - start_frame + 1) / fps, 6) if fps else None,
                "sampled_frame_indices": [int(x) for x in sampled_frame_indices],
                "num_sampled_frames": len(images),
                "caption": caption
            })

        return {
            "video_path": video_path,
            "fps": fps,
            "total_frames": total_frames,
            "scene_threshold": scene_threshold,
            "num_scenes": len(results),
            "scenes": results
        }


class LlavaSceneCaptioner:

    @staticmethod
    def _postprocess_caption(text):

        text = text.strip()

        if "[/INST]" in text:
            text = text.split("[/INST]", 1)[-1].strip()

        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:", 1)[-1].strip()

        return text

    def __init__(
        self,
        model_name="llava-hf/llava-v1.6-mistral-7b-hf",
        device="auto",
        config=LlavaSceneCaptionConfig,
    ):

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.config = config

        print("Loading LLaVA Scene Caption:", model_name)

        self.processor = LlavaNextProcessor.from_pretrained(model_name, use_fast=False)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.dtype
        ).to(device)

        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.processor.tokenizer.eos_token_id

        self.model.eval()

    def caption_frames(self, images, prompt, max_new_tokens):

        prompt = (prompt or "").strip()

        if not prompt:
            prompt = self.config.prompt

        conversation = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in images] + [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        inputs = self.processor(
            images=images,
            text=[text],
            return_tensors="pt",
            padding=True
        )

        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)

        with torch.inference_mode():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )

        generated_ids = ids[:, inputs["input_ids"].shape[1]:]
        captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return self._postprocess_caption(captions[0])

    def generate_caption(self, frames, prompt):

        if not frames:
            return ""

        return self.caption_frames(
            list(frames),
            prompt,
            self.config.max_new_tokens
        )

    def caption_video_scenes(self, video_path, prompt=None, num_frames=None, max_new_tokens=None, scene_threshold=None):

        if num_frames is None:
            num_frames = self.config.num_frames

        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens

        if scene_threshold is None:
            scene_threshold = self.config.scene_threshold

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames <= 0:
            return {
                "video_path": video_path,
                "fps": fps,
                "total_frames": total_frames,
                "scene_threshold": scene_threshold,
                "num_scenes": 0,
                "scenes": []
            }

        scene_ranges = detect_scenes(video_path, threshold=scene_threshold)
        results = []

        progress = tqdm(
            scene_ranges,
            desc=f"Captioning scenes ({Path(video_path).stem})",
            unit="scene",
            leave=False
        )

        for i, (start_frame, end_frame) in enumerate(progress):

            sampled_frame_indices = _sample_frame_indices(
                start_frame,
                end_frame,
                num_frames
            )

            images = sample_scene_frames(
                video_path,
                start_frame,
                end_frame,
                num_frames
            )

            if not images:
                continue

            caption = self.caption_frames(
                images,
                prompt,
                max_new_tokens
            )

            results.append({
                "scene_index": i,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start_sec": round(start_frame / fps, 6) if fps else None,
                "end_sec": round(end_frame / fps, 6) if fps else None,
                "duration_sec": round((end_frame - start_frame + 1) / fps, 6) if fps else None,
                "sampled_frame_indices": [int(x) for x in sampled_frame_indices],
                "num_sampled_frames": len(images),
                "caption": caption
            })

        return {
            "video_path": video_path,
            "fps": fps,
            "total_frames": total_frames,
            "scene_threshold": scene_threshold,
            "num_scenes": len(results),
            "scenes": results
        }


def process_dataset(args, dataset, config):

    if dataset == "summe":
        mapping_path = os.path.join(args.dataset_root, "summe_mapping.json")
        video_dir = os.path.join(args.dataset_root, "SumMe/videos")

    elif dataset == "tvsum":
        mapping_path = os.path.join(args.dataset_root, "tvsum_mapping.json")
        video_dir = os.path.join(args.dataset_root, "TVSum/ydata-tvsum50-v1_1/video")

    else:
        raise ValueError(dataset)

    print("Dataset:", dataset)

    with open(mapping_path) as f:
        mapping = json.load(f)

    output_dir = os.path.join(args.output_root, args.model_name, dataset)
    os.makedirs(output_dir, exist_ok=True)

    model = args.model
    keys = list(mapping.keys())

    video_progress = tqdm(
        keys,
        desc=f"Processing {dataset}",
        unit="video"
    )

    for i, key in enumerate(video_progress):

        video_name = mapping[key]
        video_path = os.path.join(video_dir, video_name + ".mp4")
        video_progress.set_postfix(video=video_name)

        result = model.caption_video_scenes(
            video_path,
            config.prompt,
            config.num_frames,
            config.max_new_tokens,
            config.scene_threshold
        )

        output = {
            "video": video_name,
            "video_path": video_path,
            "caption_model": args.caption_model,
            "model_name": args.model_name,
            "prompt": config.prompt,
            "num_frames": config.num_frames,
            "scene_threshold": config.scene_threshold,
            "fps": result["fps"],
            "total_frames": result["total_frames"],
            "num_scenes": result["num_scenes"],
            "scenes": result["scenes"]
        }

        out_file = os.path.join(output_dir, key + ".json")

        with open(out_file, "w") as f:
            json.dump(output, f, indent=2)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", default="/root/autodl-tmp/datasets")
    parser.add_argument("--output_root", default="/root/VideoSummarizationAgent/data/metadata/scene_caption")
    parser.add_argument("--datasets", nargs="+", default=["summe", "tvsum"])

    parser.add_argument("--model_name", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--caption_model", choices=["llava", "qwen"], default="qwen")
    parser.add_argument("--scene_threshold", type=float, default=24)

    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    if args.caption_model == "qwen":
        args.model_name = "Qwen/Qwen3-VL-8B-Instruct"
        if args.scene_threshold is not None:
            QwenSceneCaptionConfig.scene_threshold = args.scene_threshold
        args.model = QwenSceneCaptioner(
            model_name=args.model_name,
            device=args.device,
            config=QwenSceneCaptionConfig
        )
        args.config = QwenSceneCaptionConfig
    else:
        args.model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
        if args.scene_threshold is not None:
            LlavaSceneCaptionConfig.scene_threshold = args.scene_threshold
        args.model = LlavaSceneCaptioner(
            model_name=args.model_name,
            device=args.device,
            config=LlavaSceneCaptionConfig
        )
        args.config = LlavaSceneCaptionConfig

    for dataset in args.datasets:
        process_dataset(args, dataset, args.config)


if __name__ == "__main__":
    main()
