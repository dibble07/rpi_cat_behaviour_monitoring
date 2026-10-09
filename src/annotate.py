import json
import logging
import time
from pathlib import Path

import cv2

from config import METADATA_DIR, OUTPUT_DIR, settings
from shared import processing_busy_event, shutdown_event
from video_io import FfmpegWriter

logger = logging.getLogger(__name__)
_METADATA_DIR = Path(METADATA_DIR)
_SUMMARIES_PATH = _METADATA_DIR / "track_summaries.jsonl"


def _load_next_video_metadata() -> dict | None:
    # identify non-annotated video files
    files = {p for p in Path(OUTPUT_DIR).glob("*.mp4") if p.is_file()}
    ann = {p for p in files if "annotated" in p.name}
    ann_source = {
        p.with_name(f"{p.stem.replace('_annotated', '').replace('.tmp', '')}{p.suffix}")
        for p in ann
    }
    cand = {p for p in files if "tmp" not in p.name and "annotated" not in p.name}
    cand -= ann_source

    # remove candidates with no corresponding track summaries
    summaries = (
        list(map(json.loads, _SUMMARIES_PATH.read_text().splitlines()))
        if _SUMMARIES_PATH.exists()
        else []
    )
    summaries_by_video = {
        n: [s for s in summaries if s["video_name"] == n]
        for n in {s["video_name"] for s in summaries}
    }
    cand = {p for p in cand if p.name in summaries_by_video}

    # iterate over remaining candidates and yield the first one with complete metadata
    for video_path in sorted(cand, key=lambda p: p.name, reverse=True):

        # check metadata presence
        video_summaries = summaries_by_video[video_path.name]
        hashes_path = _METADATA_DIR / f"video-{video_path.name}.json"
        track_paths = [
            _METADATA_DIR / f"track-{r['manager_id']}-{r['track_id']}.json"
            for r in video_summaries
        ]
        if not all(path.is_file() for path in track_paths) or not hashes_path.is_file():
            continue

        # load metadata
        frame_hashes = json.loads(hashes_path.read_text())
        track_annotations = [json.loads(p.read_text()) for p in track_paths]

        return {
            "video_path": video_path,
            "hashes_path": str(hashes_path),
            "frame_hashes": frame_hashes,
            "track_paths": [str(p) for p in track_paths],
            "track_annotations": track_annotations,
            "track_summaries": video_summaries,
        }

    return None


def annotate_thread() -> None:
    """Annotate ready recordings while processing is idle"""
    logger.info("Annotation thread started")
    candidate = None

    while not shutdown_event.is_set():

        # wait for processing to be idle
        if processing_busy_event.is_set():
            time.sleep(1 / 2 / settings.FPS)
            continue

        if candidate is None:

            # load the next video metadata for annotation
            candidate = _load_next_video_metadata()
            if candidate is None:
                time.sleep(1)
                continue

            # prepare the annotation environment for the candidate video
            video_path = candidate["video_path"]
            temp_path = video_path.with_name(
                f"{video_path.stem}_annotated.tmp{video_path.suffix}"
            )
            capture = cv2.VideoCapture(str(video_path))
            writer = FfmpegWriter(
                temp_path.stem,
                capture.get(cv2.CAP_PROP_FPS),
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                settings.VIDEO_QUALITY,
            )
            candidate.update(
                {
                    "temp_path": temp_path,
                    "capture": capture,
                    "writer": writer,
                    "frame_index": 0,
                }
            )
            logger.info(f"Prepared annotation environment for video: {video_path}")

        else:

            pass

    logger.info("Annotation thread stopped")
