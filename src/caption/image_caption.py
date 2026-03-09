from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import h5py
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


class BLIP2Config:
    prompt = ""
    batch_size = 8
    max_new_tokens = 30


class LLaVAConfig:
    prompt = "Describe this frame in one sentence."
    batch_size = 24
    max_new_tokens = 80


class QwenVLConfig:
    prompt = "Describe this frame in one sentence."
    batch_size = 16
    max_new_tokens = 80


class VideoCaptionBLIP2:

    @staticmethod
    def _postprocess_caption(text, prompt):

        text = text.strip()

        if prompt:
            prompt = prompt.strip()
            if text.lower().startswith(prompt.lower()):
                text = text[len(prompt):].strip(" \n:\t")

        if "Answer:" in text:
            text = text.split("Answer:", 1)[-1].strip()

        return text

    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", device="auto"):

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32

        print("Loading BLIP2:", model_name)

        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.dtype
        ).to(device)

        self.model.eval()

    def caption_frames(self, images, prompt, max_new_tokens):

        prompt = (prompt or "").strip()

        processor_kwargs = {
            "images": images,
            "return_tensors": "pt",
            "padding": True
        }

        if prompt:
            processor_kwargs["text"] = [prompt] * len(images)

        inputs = self.processor(**processor_kwargs)

        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)

        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids = ids

        if prompt and "input_ids" in inputs:
            prompt_length = inputs["input_ids"].shape[1]
            generated_ids = ids[:, prompt_length:]

        captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        full_captions = self.processor.batch_decode(ids, skip_special_tokens=True)

        processed_captions = []

        for idx, caption in enumerate(captions):
            cleaned = self._postprocess_caption(caption, prompt)

            if not cleaned:
                cleaned = self._postprocess_caption(full_captions[idx], prompt)

            processed_captions.append(cleaned)

        return processed_captions

    def caption_video_frames(self, video_path, frame_indices, prompt, batch_size, max_new_tokens):

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        results = []

        batch_imgs = []
        batch_meta = []

        progress = tqdm(
            frame_indices,
            desc=f"Captioning frames ({Path(video_path).stem})",
            unit="frame",
            leave=False
        )

        for i, frame_idx in enumerate(progress):

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()

            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            batch_imgs.append(img)
            batch_meta.append((i, frame_idx))

            if len(batch_imgs) == batch_size:

                captions = self.caption_frames(
                    batch_imgs,
                    prompt,
                    max_new_tokens
                )

                for (seq, fidx), cap_text in zip(batch_meta, captions):
                    results.append({
                        "sequence_index": seq,
                        "frame_index": int(fidx),
                        "timestamp_sec": round(fidx / fps, 6) if fps else None,
                        "caption": cap_text
                    })

                batch_imgs = []
                batch_meta = []

        if batch_imgs:
            captions = self.caption_frames(batch_imgs, prompt, max_new_tokens)

            for (seq, fidx), cap_text in zip(batch_meta, captions):
                results.append({
                    "sequence_index": seq,
                    "frame_index": int(fidx),
                    "timestamp_sec": round(fidx / fps, 6) if fps else None,
                    "caption": cap_text
                })

        cap.release()
        return results


class VideoCaptionLLaVA:

    @staticmethod
    def _postprocess_caption(text):

        text = text.strip()

        if "[/INST]" in text:
            text = text.split("[/INST]", 1)[-1].strip()

        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:", 1)[-1].strip()

        return text

    def __init__(self, model_name="llava-hf/llava-v1.6-mistral-7b-hf", device="auto"):

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32

        print("Loading LLaVA:", model_name)

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

        if not prompt or prompt == "":
            prompt = "Describe this image."

        texts = []

        for _ in images:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True
            )

            texts.append(text)

        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True
        )

        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)

        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids = ids[:, inputs["input_ids"].shape[1]:]
        captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [self._postprocess_caption(c) for c in captions]

    def caption_video_frames(self, video_path, frame_indices, prompt, batch_size, max_new_tokens):

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        results = []

        batch_imgs = []
        batch_meta = []

        progress = tqdm(
            frame_indices,
            desc=f"Captioning frames ({Path(video_path).stem})",
            unit="frame",
            leave=False
        )

        for i, frame_idx in enumerate(progress):

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()

            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            batch_imgs.append(img)
            batch_meta.append((i, frame_idx))

            if len(batch_imgs) == batch_size:

                captions = self.caption_frames(
                    batch_imgs,
                    prompt,
                    max_new_tokens
                )

                for (seq, fidx), cap_text in zip(batch_meta, captions):
                    results.append({
                        "sequence_index": seq,
                        "frame_index": int(fidx),
                        "timestamp_sec": round(fidx / fps, 6) if fps else None,
                        "caption": cap_text
                    })

                batch_imgs = []
                batch_meta = []

        if batch_imgs:
            captions = self.caption_frames(batch_imgs, prompt, max_new_tokens)

            for (seq, fidx), cap_text in zip(batch_meta, captions):
                results.append({
                    "sequence_index": seq,
                    "frame_index": int(fidx),
                    "timestamp_sec": round(fidx / fps, 6) if fps else None,
                    "caption": cap_text
                })

        cap.release()
        return results


class VideoCaptionQwenVL:

    def __init__(self, model_name="Qwen/Qwen3-VL-8B-Instruct", device="auto"):

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32

        print("Loading QwenVL:", model_name)

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.dtype
        ).to(device)

        self.model.eval()

    def caption_frames(self, images, prompt, max_new_tokens):

        prompt = (prompt or "").strip()

        if not prompt:
            prompt = "Describe this image."

        texts = []

        for _ in images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            texts.append(text)

        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        )

        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)

        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs["input_ids"], ids)
        ]

        captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [c.strip() for c in captions]

    def caption_video_frames(self, video_path, frame_indices, prompt, batch_size, max_new_tokens):

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        results = []

        batch_imgs = []
        batch_meta = []

        progress = tqdm(
            frame_indices,
            desc=f"Captioning frames ({Path(video_path).stem})",
            unit="frame",
            leave=False
        )

        for i, frame_idx in enumerate(progress):

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()

            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            batch_imgs.append(img)
            batch_meta.append((i, frame_idx))

            if len(batch_imgs) == batch_size:

                captions = self.caption_frames(
                    batch_imgs,
                    prompt,
                    max_new_tokens
                )

                for (seq, fidx), cap_text in zip(batch_meta, captions):
                    results.append({
                        "sequence_index": seq,
                        "frame_index": int(fidx),
                        "timestamp_sec": round(fidx / fps, 6) if fps else None,
                        "caption": cap_text
                    })

                batch_imgs = []
                batch_meta = []

        if batch_imgs:
            captions = self.caption_frames(batch_imgs, prompt, max_new_tokens)

            for (seq, fidx), cap_text in zip(batch_meta, captions):
                results.append({
                    "sequence_index": seq,
                    "frame_index": int(fidx),
                    "timestamp_sec": round(fidx / fps, 6) if fps else None,
                    "caption": cap_text
                })

        cap.release()
        return results


def process_dataset(args, dataset, config):

    if dataset == "summe":
        h5_path = os.path.join(args.dataset_root, "eccv16_dataset_summe_google_pool5.h5")
        mapping_path = os.path.join(args.dataset_root, "summe_mapping.json")
        video_dir = os.path.join(args.dataset_root, "SumMe/videos")

    elif dataset == "tvsum":
        h5_path = os.path.join(args.dataset_root, "eccv16_dataset_tvsum_google_pool5.h5")
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

    with h5py.File(h5_path, "r") as h5:

        keys = list(h5.keys())

        video_progress = tqdm(
            keys,
            desc=f"Processing {dataset}",
            unit="video"
        )

        for i, key in enumerate(video_progress):

            video_name = mapping[key]
            video_path = os.path.join(video_dir, video_name + ".mp4")
            video_progress.set_postfix(video=video_name)

            group = h5[key]
            frame_indices = group["picks"][()].tolist()

            captions = model.caption_video_frames(
                video_path,
                frame_indices,
                config.prompt,
                config.batch_size,
                config.max_new_tokens
            )

            output = {
                "video": video_name,
                "captions": captions
            }

            out_file = os.path.join(output_dir, key + ".json")

            with open(out_file, "w") as f:
                json.dump(output, f, indent=2)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", default="/root/autodl-tmp/datasets")
    parser.add_argument("--output_root", default="/root/VideoSummarizationAgent/data/metadata/image_caption")
    parser.add_argument("--datasets", nargs="+", default=["summe", "tvsum"])

    parser.add_argument("--model_name", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--caption_model", choices=["blip2", "llava", "qwen"], default="qwen")

    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    if args.caption_model == "blip2":
        args.model_name = "Salesforce/blip2-opt-2.7b"
        args.model = VideoCaptionBLIP2(
            model_name=args.model_name,
            device=args.device
        )
        args.config = BLIP2Config
    elif args.caption_model == "qwen":
        args.model_name = "Qwen/Qwen3-VL-8B-Instruct"
        args.model = VideoCaptionQwenVL(
            model_name=args.model_name,
            device=args.device
        )
        args.config = QwenVLConfig
    else:
        args.model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
        args.model = VideoCaptionLLaVA(
            model_name=args.model_name,
            device=args.device
        )
        args.config = LLaVAConfig

    for dataset in args.datasets:
        process_dataset(args, dataset, args.config)


if __name__ == "__main__":
    main()