from __future__ import annotations

THEME_CHOICES = ["story", "tutorial", "landscape", "sports", "vlog", "general"]


PLANNER_SYSTEM_PROMPT = """
You are the planner of a video summarization system.
You must return JSON only.
All scores must be floats in [0, 1].
The expert weights should sum to 1.
The video_theme must be one of: story, tutorial, landscape, sports, vlog, general.
""".strip()


EXPERT_SYSTEM_PROMPT_TEMPLATE = """
You are the {agent_name} for a video summarization system.
You only evaluate one dimension.
Return JSON only with keys: score, reason.
The score must be a float in [0, 1].
""".strip()


CAPTION_SYSTEM_PROMPT = """
You are a video captioning assistant.
Describe the content shown in sampled frames of one short video segment.
Return one concise paragraph.
""".strip()


def build_planner_plan_user_prompt(segment_captions: list[str]) -> str:
    captions_text = "\n".join(
        f"Segment {index}: {caption}" for index, caption in enumerate(segment_captions)
    )
    return (
        "Analyze the following segment captions for the whole video.\n"
        "Infer the video theme, summarize the whole video briefly, and assign expert weights.\n"
        "Return JSON with keys: video_theme, global_summary, expert_weights, reason.\n\n"
        f"Segment captions:\n{captions_text}"
    )


def build_planner_segment_user_prompt(
    video_theme: str,
    global_summary: str,
    current_caption: str,
    memory_context: str,
) -> str:
    return (
        "Evaluate the importance of the current segment for the final video summary.\n"
        "Return JSON with keys: score, reason.\n\n"
        f"Video theme: {video_theme}\n"
        f"Global summary: {global_summary}\n"
        f"Memory context:\n{memory_context or 'None'}\n"
        f"Current segment caption: {current_caption}"
    )


def build_expert_user_prompt(
    dimension_desc: str,
    video_theme: str,
    global_summary: str,
    current_caption: str,
    memory_context: str,
) -> str:
    return (
        f"Evaluate the current segment only from this dimension: {dimension_desc}.\n"
        "Return JSON with keys: score, reason.\n\n"
        f"Video theme: {video_theme}\n"
        f"Global summary: {global_summary}\n"
        f"Memory context:\n{memory_context or 'None'}\n"
        f"Current segment caption: {current_caption}"
    )


def build_caption_user_prompt(frame_descriptions: list[str]) -> str:
    descriptions = "\n".join(f"- {text}" for text in frame_descriptions)
    return (
        "The following texts describe sampled frames from one video segment.\n"
        "Write one concise segment caption based on them.\n\n"
        f"Frame descriptions:\n{descriptions}"
    )