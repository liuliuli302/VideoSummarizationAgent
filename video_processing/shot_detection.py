"""Scene detection based on PySceneDetect."""

from __future__ import annotations

from typing import List, Tuple

import cv2
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector


def detect_scenes(video_path: str) -> List[Tuple[int, int]]:
    """Detect scenes in a video using PySceneDetect's `ContentDetector`.

    Returns:
        A list of `(start_frame, end_frame)` tuples.
    """
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())

    scene_manager.detect_scenes(video=video, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    if scene_list:
        return [(int(start.get_frames()), int(end.get_frames())) for start, end in scene_list]

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Failed to open video file for scene fallback: {video_path}")
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    return [(0, max(1, total_frames))]