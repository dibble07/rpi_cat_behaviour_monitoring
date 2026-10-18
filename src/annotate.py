import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from config import CAT_COLOUR_MAP, METADATA_DIR, OBJECT_COLOUR_MAP, OUTPUT_DIR, settings
from shared import processing_busy_event, processing_stopped_event
from video_io import FfmpegWriter

logger = logging.getLogger(__name__)
_METADATA_DIR = Path(METADATA_DIR)
_SUMMARIES_PATH = _METADATA_DIR / "track_summaries.json"
_ANN_FAILS: defaultdict[str, int] = defaultdict(int)


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
        json.loads(_SUMMARIES_PATH.read_text()) if _SUMMARIES_PATH.exists() else []
    )
    summaries_by_video = {
        n: [s for s in summaries if s["video_name"] == n]
        for n in {s["video_name"] for s in summaries}
    }
    cand = {p for p in cand if p.name in summaries_by_video and _ANN_FAILS[p.name] < 5}

    # iterate over remaining candidates and yield the first one with complete metadata
    for video_path in sorted(
        cand, key=lambda p: (_ANN_FAILS[p.name], -int(p.stem.replace("_", "")))
    ):

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


def _get_abandoned_annotation_files() -> list[Path]:
    return [p for p in Path(OUTPUT_DIR).glob("*_annotated.tmp.mp4") if p.is_file()]


def _annotate_frame(
    frame: np.ndarray,
    frame_hash: str,
    track_annotations: list[dict[str, dict[str, object]]],
    track_summaries: list[dict],
) -> None:
    height, width = frame.shape[:2]
    for annotations, summary in zip(track_annotations, track_summaries):
        colour = CAT_COLOUR_MAP.get(
            summary["cat_id"],
            OBJECT_COLOUR_MAP.get(summary["object_name"], (200, 200, 200)),
        )
        if annotation := annotations.get(frame_hash):
            bbox = annotation["bbox"]
            x_center, y_center, box_width, box_height = map(float, bbox)  # type: ignore[call-overload]
            x1 = max(0, int((x_center - box_width / 2) * width))
            y1 = max(0, int((y_center - box_height / 2) * height))
            x2 = min(width - 1, int((x_center + box_width / 2) * width))
            y2 = min(height - 1, int((y_center + box_height / 2) * height))
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                colour,
                max(2, min(width, height) // 250),
            )
            if behaviour := annotation.get("behaviour"):
                text = str(behaviour)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = max(0.5, min(width, height) / 1200)
                thickness = max(1, min(width, height) // 500)
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, font, font_scale, thickness
                )
                padding = max(2, thickness * 2)
                label_x = min(x1, max(0, width - text_width - 2 * padding))
                label_height = text_height + baseline + 2 * padding
                label_y = (
                    y1 - label_height
                    if y1 >= label_height
                    else min(height - label_height, y2 + 1)
                )
                label_y = max(0, label_y)
                cv2.rectangle(
                    frame,
                    (label_x, label_y),
                    (
                        min(width - 1, label_x + text_width + 2 * padding),
                        min(height - 1, label_y + label_height),
                    ),
                    colour,
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    text,
                    (label_x + padding, label_y + padding + text_height),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA,
                )


def _is_valid_video(video_path: Path, temp_path: Path) -> bool:
    video_is_file, temp_is_file = video_path.is_file(), temp_path.is_file()
    if valid := video_is_file and temp_is_file:
        video = cv2.VideoCapture(str(video_path))
        temp = cv2.VideoCapture(str(temp_path))
        video_is_open, temp_is_open = video.isOpened(), temp.isOpened()
        if valid := video_is_open and temp_is_open:
            for attr in [
                cv2.CAP_PROP_FRAME_COUNT,
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS,
            ]:
                video_val, temp_val = video.get(attr), temp.get(attr)
                if not (valid := video_val == temp_val):
                    logger.warning(
                        f"Mismatched {attr}: video={video_val}, temp={temp_val}"
                    )
                    break
        else:
            logger.warning(f"Files open: video={video_is_open}, temp={temp_is_open}")
        video.release()
        temp.release()
    else:
        logger.warning(f"Files exist: video={video_is_file}, temp={temp_is_file}")
    return valid


def _close_annotation(candidate: dict) -> None:

    # close the video capture and writer
    candidate["capture"].release()
    candidate["writer"].release()
    temp = candidate["temp_path"]
    annotated = temp.with_name(f"{temp.stem.replace('.tmp', '')}{temp.suffix}")

    if _is_valid_video(candidate["video_path"], temp):

        # rename temp file
        os.replace(temp, annotated)
        logger.info(f"Final annotated video saved: {annotated}")

        # delete metadata files
        deleted_metadata = candidate["track_paths"] + [candidate["hashes_path"]]
        for path in deleted_metadata:
            Path(path).unlink()
        logger.info(f"Deleted metadata files: {', '.join(deleted_metadata)}")

    else:

        # delete invalid annotated video
        video_name = candidate["video_path"].name
        _ANN_FAILS[video_name] += 1
        logger.warning(
            f"Annotated video is invalid: {temp} ({_ANN_FAILS[video_name]} attempts)"
        )
        temp.unlink(missing_ok=True)


def annotate_thread() -> None:
    """Annotate ready recordings while processing is idle"""
    logger.info("Annotation thread started")
    candidate = None
    annotations_complete = False

    while not processing_stopped_event.is_set() or not annotations_complete:

        # wait for processing to be idle
        if processing_busy_event.is_set():
            time.sleep(1 / 2 / settings.FPS)
            continue

        if candidate is None:

            # clean up any abandoned annotation files
            for temp_path in _get_abandoned_annotation_files():
                video_name = temp_path.name.removesuffix("_annotated.tmp.mp4") + ".mp4"
                _ANN_FAILS[video_name] += 1
                logger.warning(
                    f"Deleting abandoned annotated video: {temp_path} ({_ANN_FAILS[video_name]} attempts)"
                )
                temp_path.unlink()
            if processing_busy_event.is_set():
                continue

            # load the next video metadata for annotation
            processing_stopped = processing_stopped_event.is_set()
            candidate = _load_next_video_metadata()
            annotations_complete = processing_stopped and candidate is None
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
                temp_path,
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
            logger.info(
                f"Prepared annotation environment for video: {video_path} ({_ANN_FAILS[video_path.name]} failed attempts); selected tracks: {candidate["track_paths"]}"
            )

        else:

            # load next frame
            success, frame = candidate["capture"].read()

            # annotate frame
            if success:
                frame_hash = candidate["frame_hashes"][candidate["frame_index"]]
                lg = logger.info if candidate["frame_index"] % 10 == 0 else logger.debug
                lg(
                    f"Annotating frame {candidate['frame_index']} ({frame_hash}) of {candidate['video_path']}"
                )
                _annotate_frame(
                    frame,
                    frame_hash,
                    candidate["track_annotations"],
                    candidate["track_summaries"],
                )
                candidate["writer"].write(frame, frame_hash)
                candidate["frame_index"] += 1

            # close annotation for the current video
            else:
                if candidate["frame_index"] == 0:
                    logger.warning(
                        f"Failed to read first frame of {candidate['video_path']}"
                    )
                _close_annotation(candidate)
                candidate = None

            # throttle the annotation loop
            time.sleep(1 / 10 / settings.FPS)

    logger.info("Annotation thread stopped")
