from __future__ import annotations

import os

import torch
from src.data.schemas import Segment, SegmentCaption
from src.llm.client import LLMClient
from src.preprocessing.video_reader import read_frames
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class SegmentCaptioner:
    _shared_processor = None
    _shared_model = None
    _shared_model_name: str | None = None
    _shared_device: str | None = None

    def __init__(self, llm_client: LLMClient, fallback_text: str = "A short video segment.") -> None:
        self.llm_client = llm_client
        self.fallback_text = fallback_text
        self.model_name = os.getenv("SEGMENT_CAPTION_MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
        self.device = os.getenv("SEGMENT_CAPTION_DEVICE", "auto")
        self.max_new_tokens = int(os.getenv("SEGMENT_CAPTION_MAX_NEW_TOKENS", "96"))
        self.prompt = os.getenv(
            "SEGMENT_CAPTION_PROMPT",
            "Describe the sampled frames as one concise segment caption. Focus on persistent actions, scene context, and the main event.",
        )

    def caption_segment(self, video_path: str, segment: Segment) -> SegmentCaption:
        images = read_frames(video_path=video_path, frame_indices=segment.caption_frame_indices)
        if not images:
            caption = self.fallback_text
        else:
            caption = self._generate_local_caption(images)

        return SegmentCaption(
            segment_id=segment.segment_id,
            start_frame=segment.start_frame,
            end_frame=segment.end_frame,
            caption_frame_indices=segment.caption_frame_indices,
            caption=caption.strip() or self.fallback_text,
        )

    def _generate_local_caption(self, images) -> str:
        processor, model, device = self._get_or_load_qwen_vl()
        conversation = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(device)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        generated_ids = [
            full_output[len(input_ids):]
            for input_ids, full_output in zip(inputs["input_ids"], output_ids)
        ]
        captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        caption = captions[0].strip() if captions else ""
        return caption or self.fallback_text

    def _get_or_load_qwen_vl(self):
        resolved_device = self._resolve_device(self.device)
        if (
            SegmentCaptioner._shared_model is None
            or SegmentCaptioner._shared_processor is None
            or SegmentCaptioner._shared_model_name != self.model_name
            or SegmentCaptioner._shared_device != resolved_device
        ):
            torch_dtype = torch.float16 if resolved_device.startswith("cuda") else torch.float32
            SegmentCaptioner._shared_processor = AutoProcessor.from_pretrained(self.model_name)
            SegmentCaptioner._shared_model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
            ).to(resolved_device)
            SegmentCaptioner._shared_model.eval()
            SegmentCaptioner._shared_model_name = self.model_name
            SegmentCaptioner._shared_device = resolved_device

        return (
            SegmentCaptioner._shared_processor,
            SegmentCaptioner._shared_model,
            SegmentCaptioner._shared_device,
        )

    def _resolve_device(self, requested_device: str) -> str:
        if requested_device != "auto":
            return requested_device
        return "cuda" if torch.cuda.is_available() else "cpu"