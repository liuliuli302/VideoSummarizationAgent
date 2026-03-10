from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm


load_dotenv()


DEFAULT_VIDEO_SUMMARIZATION_PROMPT_TEMPLATE = (
	"""
You are a video understanding assistant.

The following are scene-level captions extracted from a video.
Each caption describes one scene in chronological order.

Your task is to summarize these scene captions into ONE concise video-level caption that describes the overall content of the video.

Requirements:
1. Preserve the main events and storyline of the video.
2. Follow the temporal order of scenes.
3. Remove redundant or repeated details.
4. Produce a coherent and natural description.
5. The final caption should be 1–2 paragraph.

Scene Captions:
{scene_captions}

Video Caption:
"""
)


class VideoCaptionSummarizerConfig:
	prompt_template = DEFAULT_VIDEO_SUMMARIZATION_PROMPT_TEMPLATE
	max_tokens = 240
	max_scene_caption_chars = 100000
	max_total_input_chars = 24000
	base_url = "https://www.dmxapi.cn/v1"


def _compact_model_dir_name(model_name: str | None) -> str:

	model_name = str(model_name or "").strip()

	if not model_name:
		return "unknown"

	return Path(model_name).name


def _default_source_model_name(source_caption_model: str) -> str:

	if source_caption_model == "qwen":
		return "Qwen/Qwen3-VL-8B-Instruct"

	if source_caption_model == "llava":
		return "llava-hf/llava-v1.6-mistral-7b-hf"

	return "DeepSeek-V3.2"


def _get_scene_caption_input_dir(args: argparse.Namespace, dataset: str) -> str:

	source_model_name = args.source_model_name or _default_source_model_name(
		args.source_caption_model
	)

	if args.source_caption_model == "llm":
		return os.path.join(
			args.scene_caption_root,
			"llm",
			_compact_model_dir_name(args.image_caption_model),
			_compact_model_dir_name(source_model_name),
			dataset,
		)

	return os.path.join(
		args.scene_caption_root,
		_compact_model_dir_name(source_model_name),
		dataset,
	)


def _get_video_caption_output_dir(args: argparse.Namespace, dataset: str) -> str:

	source_model_name = args.source_model_name or _default_source_model_name(
		args.source_caption_model
	)

	if args.source_caption_model == "llm":
		return os.path.join(
			args.output_root,
			"llm",
			_compact_model_dir_name(args.image_caption_model),
			_compact_model_dir_name(source_model_name),
			_compact_model_dir_name(args.model_name),
			dataset,
		)

	return os.path.join(
		args.output_root,
		_compact_model_dir_name(source_model_name),
		_compact_model_dir_name(args.model_name),
		dataset,
	)


class VideoCaptionSummarizer:

	def __init__(
		self,
		model_name: str = "DeepSeek-V3.2",
		base_url: str | None = None,
		api_key: str | None = None,
		config: type[VideoCaptionSummarizerConfig] = VideoCaptionSummarizerConfig,
	) -> None:

		api_key = api_key or os.getenv("OPENAI_API_KEY")
		base_url = base_url or os.getenv("OPENAI_BASE_URL") or config.base_url

		if not api_key:
			raise RuntimeError("OPENAI_API_KEY not found.")

		client_kwargs: dict[str, Any] = {"api_key": api_key}

		if base_url:
			client_kwargs["base_url"] = base_url

		self.client = OpenAI(**client_kwargs)
		self.model_name = model_name
		self.base_url = base_url
		self.config = config

		print("Loading Video Caption Summarizer:", model_name)

	@staticmethod
	def load_scene_caption_json(json_path: str) -> dict[str, Any]:

		with open(json_path) as f:
			return json.load(f)

	@staticmethod
	def collect_scene_captions(data: dict[str, Any]) -> list[dict[str, Any]]:

		scenes = data.get("scenes", [])

		if not isinstance(scenes, list):
			return []

		valid_scenes = [item for item in scenes if isinstance(item, dict)]
		return sorted(valid_scenes, key=lambda item: item.get("scene_index", 0))

	@staticmethod
	def _shorten_text(text: str, max_chars: int) -> str:

		text = " ".join(str(text or "").strip().split())

		if max_chars <= 0 or len(text) <= max_chars:
			return text

		return text[: max(0, max_chars - 3)].rstrip() + "..."

	def build_prompt(self, scene_captions: list[dict[str, Any]]) -> str:

		lines: list[str] = []
		total_chars = 0

		for item in scene_captions:
			caption = self._shorten_text(
				str(item.get("caption", "")),
				self.config.max_scene_caption_chars,
			)

			if not caption:
				continue

			scene_index = int(item.get("scene_index", len(lines))) + 1
			start_sec = item.get("start_sec")
			end_sec = item.get("end_sec")

			if start_sec is not None and end_sec is not None:
				prefix = f"Scene {scene_index} ({start_sec:.2f}-{end_sec:.2f}s): "
			else:
				prefix = f"Scene {scene_index}: "

			line = prefix + caption

			if lines and total_chars + len(line) > self.config.max_total_input_chars:
				remaining = len(scene_captions) - len(lines)
				lines.append(f"... {remaining} additional scenes omitted due to prompt length.")
				break

			lines.append(line)
			total_chars += len(line)

		scene_caption_text = "\n".join(lines)
		return self.config.prompt_template.format(scene_captions=scene_caption_text)

	def summarize_video(self, scene_caption_json_path: str) -> dict[str, Any]:

		data = self.load_scene_caption_json(scene_caption_json_path)
		scene_captions = self.collect_scene_captions(data)

		if not scene_captions:
			return {
				"video": data.get("video"),
				"video_path": data.get("video_path"),
				"scene_caption_json_path": scene_caption_json_path,
				"num_scenes": 0,
				"caption": "",
			}

		prompt = self.build_prompt(scene_captions)
		response = self.client.chat.completions.create(
			model=self.model_name,
			messages=[{"role": "user", "content": prompt}],
			# max_tokens=self.config.max_tokens,
		)

		content = response.choices[0].message.content

		return {
			"video": data.get("video"),
			"video_path": data.get("video_path"),
			"scene_caption_json_path": scene_caption_json_path,
			"num_scenes": len(scene_captions),
			"caption": content.strip() if content else "",
		}


def process_dataset(
	args: argparse.Namespace,
	dataset: str,
	config: type[VideoCaptionSummarizerConfig],
) -> None:

	input_dir = _get_scene_caption_input_dir(args, dataset)
	output_dir = _get_video_caption_output_dir(args, dataset)

	if not os.path.isdir(input_dir):
		print("Skip missing scene caption directory:", input_dir)
		return

	os.makedirs(output_dir, exist_ok=True)

	json_files = sorted(Path(input_dir).glob("*.json"))

	if not json_files:
		print("Skip empty scene caption directory:", input_dir)
		return

	print("Dataset:", dataset)
	print("Scene caption input:", input_dir)
	print("Video caption output:", output_dir)

	progress = tqdm(json_files, desc=f"Summarizing videos ({dataset})", unit="video")

	for json_file in progress:
		out_file = Path(output_dir) / json_file.name

		if out_file.exists() and not args.overwrite:
			continue

		result = args.model.summarize_video(str(json_file))

		output = {
			"video": result["video"],
			"video_path": result["video_path"],
			"scene_caption_json_path": result["scene_caption_json_path"],
			"source_caption_model": args.source_caption_model,
			"source_model_name": args.source_model_name,
			"image_caption_model": (
				args.image_caption_model if args.source_caption_model == "llm" else None
			),
			"video_caption_model": args.model_name,
			"video_prompt_template": config.prompt_template,
			"max_tokens": config.max_tokens,
			"num_scenes": result["num_scenes"],
			"caption": result["caption"],
		}

		with open(out_file, "w") as f:
			json.dump(output, f, indent=2, ensure_ascii=False)


def main() -> None:

	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--scene_caption_root",
		default="/root/VideoSummarizationAgent/data/metadata/scene_caption",
	)
	parser.add_argument(
		"--output_root",
		default="/root/VideoSummarizationAgent/data/metadata/video_caption",
	)
	parser.add_argument("--datasets", nargs="+", default=["tvsum"])

	parser.add_argument("--model_name", default="DeepSeek-V3.2")
	parser.add_argument(
		"--source_caption_model",
		choices=["llava", "qwen", "llm"],
		default="qwen",
	)
	parser.add_argument("--source_model_name", default=None)
	parser.add_argument("--image_caption_model", default="Qwen3-VL-8B-Instruct")
	parser.add_argument("--llm_base_url", default=None)
	parser.add_argument("--overwrite", action="store_true")

	args = parser.parse_args()

	args.source_model_name = (
		args.source_model_name or _default_source_model_name(args.source_caption_model)
	)
	args.model = VideoCaptionSummarizer(
		model_name=args.model_name,
		base_url=args.llm_base_url,
		config=VideoCaptionSummarizerConfig,
	)
	args.config = VideoCaptionSummarizerConfig

	os.makedirs(args.output_root, exist_ok=True)

	for dataset in args.datasets:
		process_dataset(args, dataset, args.config)


if __name__ == "__main__":
	main()
