from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PERSON_LABEL_ID = "not a cat"


def setup_notebook() -> Path:
    """Set up shared project paths and return the repository root"""
    notebook_dir = Path(__file__).resolve().parent
    project_root = notebook_dir.parent
    os.chdir(project_root)
    for path in (project_root / "src", notebook_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    return project_root


class VideoInfo:
    """Provide video metadata, cached frames, and frame hashes"""

    def __init__(self, path: str | Path) -> None:
        self.video_path = Path(path)
        self.video_name = self.video_path.name
        self._images: list[np.ndarray] | None = None

        capture = cv2.VideoCapture(str(self.video_path))
        try:
            self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        if self.fps <= 0:
            raise ValueError(f"Unable to read video FPS from {self.video_path}")

    def __len__(self) -> int:
        return self.frame_count

    @property
    def images(self) -> list[np.ndarray]:
        """Load and cache all decodable video frames"""
        if self._images is None:
            capture = cv2.VideoCapture(str(self.video_path))
            try:
                images = []
                while True:
                    success, image = capture.read()
                    if not success:
                        break
                    images.append(image)
                self._images = images
            finally:
                capture.release()
        return self._images

    @property
    def hashes(self) -> tuple[str, ...]:
        """Return the cached hashes for this video's frames"""
        from utils import get_video_hashes

        return get_video_hashes(self.video_path)

    def close(self) -> None:
        """Release cached video frames"""
        self._images = None


def is_video_annotated(video: VideoInfo, annotations: list[dict]) -> bool:
    """Check that a video has annotations without overlapping windows"""
    video_annotations = [
        row for row in annotations if row["video_name"] == video.video_name
    ]
    if not video_annotations:
        return False

    hash_to_index = {frame_hash: index for index, frame_hash in enumerate(video.hashes)}
    windows = defaultdict(list)
    for row in video_annotations:
        start = hash_to_index[row["frame_hash_start"]]
        end = hash_to_index[row["frame_hash_end"]]
        windows[row["cat_id"]].append((start, end))

    for cat_id, intervals in windows.items():
        intervals.sort()
        if any(
            current[0] <= previous[1]
            for previous, current in zip(intervals, intervals[1:])
        ):
            raise ValueError(f"{video.video_name} annotations overlap for {cat_id}")
    return True


def load_annotations(path: str | Path) -> list[dict]:
    """Load behaviour annotations from a JSON file"""
    return json.loads(Path(path).read_text())


def annotation_intervals(
    annotations: list[dict], frame_hashes: tuple[str, ...], video_name: str
) -> list[tuple[str, int, int, str | None]]:
    """Convert matching annotation hash boundaries to frame intervals"""
    hash_to_index = {h: i for i, h in enumerate(frame_hashes)}
    intervals = []
    for annotation in annotations:
        if annotation["video_name"] != video_name:
            continue
        intervals.append(
            (
                annotation["cat_id"],
                hash_to_index[annotation["frame_hash_start"]],
                hash_to_index[annotation["frame_hash_end"]],
                annotation["behaviour"],
            )
        )
    return intervals
