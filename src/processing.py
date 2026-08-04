from __future__ import annotations

import hashlib
import logging
import os
import queue
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

import utils
from config import settings
from shared import frame_queue, shutdown_event
from tracking import TrackFrame, TrackManager, TrackState, TrackSummary

logger = logging.getLogger(__name__)

# annotation font
FONT = cv2.FONT_HERSHEY_SIMPLEX

# load object detection model
MODEL = YOLO(Path("models") / settings.MODEL_DETECTION_PATH, task="detect")
_ = MODEL(
    np.zeros((settings.FRAME_HEIGHT, settings.FRAME_WIDTH, 3), dtype=np.uint8),
    imgsz=settings.DETECTION_IMGSZ,
    conf=settings.CONF,
    iou=settings.NMS_IOU_THRESHOLD,
    verbose=False,
    max_det=settings.MAX_DETS,
)

# define background subtractor
BACK_SUB = cv2.createBackgroundSubtractorMOG2(
    history=settings.BACKGROUND_HISTORY, detectShadows=False
)

# low-resolution dimensions used for motion detection
_GREY_W = 640
_GREY_H = 480


class FFmpegWriter:
    """Drop-in replacement for cv2.VideoWriter using ffmpeg for quality-controlled MJPEG."""

    def __init__(
        self,
        path: str,
        fps: float,
        width: int,
        height: int,
        qv: int,
    ):
        self.output_path = path
        cmd = [
            "ffmpeg",
            "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-c:v",
            "mjpeg",
            "-q:v",
            str(qv),
            "-pix_fmt",
            "yuvj420p",
            path,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self._proc.poll() is not None:
            raise RuntimeError(f"ffmpeg failed to start for {path}")

    def write(self, frame: np.ndarray) -> None:
        self._proc.stdin.write(frame.tobytes())

    def release(self) -> None:
        self._proc.stdin.close()
        self._proc.wait()


class Frame:
    """Store frame image and timestamp as well as supplementary processing and annotations"""

    def __init__(
        self,
        timestamp: datetime,
        image: np.ndarray,
        prev_frame: Optional[Frame],
        prev_track_mask: np.ndarray,
        forced_detection_run: bool,
    ) -> None:
        self.timestamp = timestamp
        self.image = np.ascontiguousarray(image)
        self.prev_track_mask = prev_track_mask
        self.forced_detection_run = forced_detection_run
        start = datetime.now()
        self.image_grey_blur = cv2.GaussianBlur(
            cv2.resize(
                cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), (_GREY_W, _GREY_H)
            ),
            (5, 5),
            0,
        )
        self.hash = hashlib.md5(self.image_grey_blur.tobytes()).hexdigest()[:6]
        utils.log_timing(logger, "Blur and hash", start, self.hash)

        if prev_frame is None:
            logger.warning(f"({self.hash}) No previous frame provided")
            self.prev_image_grey_blur = np.zeros((_GREY_H, _GREY_W), dtype=np.uint8)
        else:
            self.prev_image_grey_blur = prev_frame.image_grey_blur.copy().astype(
                np.uint8
            )

    def _identify_search_area(self):
        """Identify search area via frame differencing, background subtraction and previous tracks."""
        # start timing
        start = datetime.now()
        logger.debug(f"({self.hash}) Running motion detection")

        # calculate mask of changes from the previous frame
        diff = cv2.absdiff(self.prev_image_grey_blur, self.image_grey_blur)
        _, diff_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # get mask of foreground from background removal model
        fore_mask = BACK_SUB.apply(self.image_grey_blur)

        # combine change and foreground masks
        motion_mask = cv2.bitwise_or(diff_mask, fore_mask)

        # remove small pixel clusters
        contours, _ = cv2.findContours(
            motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in contours:
            if cv2.contourArea(c) < int(0.00005 * motion_mask.size):
                cv2.drawContours(motion_mask, [c], -1, (0.0,), -1)

        # resize previous track mask
        track_mask = cv2.resize(
            self.prev_track_mask,
            (_GREY_W, _GREY_H),
            interpolation=cv2.INTER_NEAREST,
        )

        # combine motion and previous detection masks into the next detection region
        search_mask = cv2.bitwise_or(motion_mask, track_mask)

        # store search mask and presence flag
        self._search_mask = cv2.resize(
            search_mask,
            (settings.FRAME_WIDTH, settings.FRAME_HEIGHT),
            interpolation=cv2.INTER_NEAREST,
        )
        self._has_search_area = (
            cv2.countNonZero(search_mask) / search_mask.size
        ) > 0.001

        # log detection duration
        utils.log_timing(logger, "Motion detection", start, self.hash)

        # log motion
        if self._has_search_area:
            logger.debug(f"({self.hash}) Has search area: {self._has_search_area}")

    @property
    def search_mask(self) -> np.ndarray:
        if not hasattr(self, "_search_mask"):
            self._identify_search_area()
        return self._search_mask

    @property
    def has_search_area(self) -> bool:
        if not hasattr(self, "_has_search_area"):
            self._identify_search_area()
        return self._has_search_area

    def _identify_search_bbox(self):

        logger.debug(f"({self.hash}) Identifying search bounding box")

        # identify bbox of the next detection region
        h, w = self.image.shape[:2]
        ys, xs = np.where(self.search_mask > 0)
        x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
        self._search_bbox = utils.expand_bbox_from_bounds(
            x_min, x_max, y_min, y_max, w, h, 0.1
        )

    @property
    def search_bbox(self) -> list:
        if not hasattr(self, "_search_bbox"):
            self._identify_search_bbox()
        return self._search_bbox

    def detect_objects(self) -> tuple[List[TrackFrame], bool]:
        track_frames = []

        # run detection if motion is present, active tracks exist, or on forced cadence
        if self.has_search_area or self.forced_detection_run:

            # log forced run
            if not self.has_search_area and self.forced_detection_run:
                logger.debug(f"({self.hash}) Forced object detection run")

            # start timing
            start = datetime.now()
            logger.debug(f"({self.hash}) Running object detection")
            did_run_detection = True

            # crop image to the search region
            if not self.has_search_area:
                image = self.image.copy()
                offsets = np.array([0, 0, 0, 0], dtype=np.int32)
            else:
                image = self.image[
                    self.search_bbox[1] : self.search_bbox[3],
                    self.search_bbox[0] : self.search_bbox[2],
                ].copy()
                offsets = np.array(
                    [
                        self.search_bbox[0],
                        self.search_bbox[1],
                        self.search_bbox[0],
                        self.search_bbox[1],
                    ],
                    dtype=np.int32,
                )

            # run model inference
            results = MODEL(
                image,
                imgsz=settings.DETECTION_IMGSZ,
                conf=settings.CONF,
                iou=settings.NMS_IOU_THRESHOLD,
                verbose=False,
                max_det=settings.MAX_DETS,
            )[0]

            # process detections
            for r in results.boxes:
                bbox = tuple(r.xyxy[0].cpu().numpy().astype(np.int32) + offsets)
                object_name = MODEL.names[int(r.cls[0].item())]
                frame_wh = (self.image.shape[1], self.image.shape[0])
                track_frames.append(
                    TrackFrame(
                        frame_hash=self.hash,
                        image=self.image,
                        bbox=utils.Bbox(xyxy=bbox, frame_wh=frame_wh),
                        object_name=object_name,
                        confidence=float(r.conf[0].item()),
                    )
                )

            # log detection duration
            utils.log_timing(logger, "Object detection", start, self.hash)

            # log detections
            if track_frames:
                logger.debug(
                    f"({self.hash}) Object(s) detected: {', '.join(f'{f.object_name} ({f.confidence:.2f})' for f in track_frames)}"
                )
        else:
            did_run_detection = False
        return track_frames, did_run_detection

    @property
    def processing_track_summaries(self) -> List[TrackSummary]:
        if not hasattr(self, "_processing_track_summaries"):
            raise RuntimeError(
                f"Processing track summaries not set yet for frame {self.hash}"
            )
        return self._processing_track_summaries

    @processing_track_summaries.setter
    def processing_track_summaries(
        self, processing_track_summaries: List[TrackSummary]
    ) -> None:
        self._processing_track_summaries = processing_track_summaries

    @property
    def recording_track_summaries(self) -> List[TrackSummary]:
        if not hasattr(self, "_recording_track_summaries"):
            raise RuntimeError(
                f"Recording track summaries not set yet for frame {self.hash}"
            )
        return self._recording_track_summaries

    @recording_track_summaries.setter
    def recording_track_summaries(
        self, recording_track_summaries: List[TrackSummary]
    ) -> None:
        self._recording_track_summaries = recording_track_summaries

    @property
    def image_annotated(self) -> np.ndarray:
        if not hasattr(self, "_image_annotated"):

            # start timing
            start = datetime.now()
            logger.debug(f"({self.hash}) Annotating image")

            # copy image ready to be annotated
            self._image_annotated = self.image.copy()

            # draw frame hash label in top-left corner
            (txt_w, txt_h), txt_baseline = cv2.getTextSize(self.hash, FONT, 1, 1)
            box_bot_right = (int(txt_w * 1.1), int(txt_h * 1.2 + txt_baseline))
            cv2.rectangle(self._image_annotated, (0, 0), box_bot_right, (0, 0, 0), -1)
            cv2.putText(
                self._image_annotated,
                self.hash,
                (int(txt_w * 0.05), int(txt_h * 1.1 + txt_baseline)),
                FONT,
                1,
                (255, 255, 255),
            )

            # annotate using track summaries
            r_summaries_map = {
                s.track_id: s.state for s in self.recording_track_summaries
            }
            for summary in sorted(
                [
                    s
                    for s in self.processing_track_summaries
                    if r_summaries_map[s.track_id]
                    in [TrackState.ACTIVE, TrackState.STALE]
                ],
                key=lambda s: (s.state, s.track_id),
                reverse=True,
            ):
                track_frame = summary.last_valid_frame

                # unpack box coords
                x1, y1, x2, y2 = track_frame.bbox.xyxy
                ann_colour = utils.CAT_COLOUR_MAP.get(
                    summary.cat_name, utils.OBJECT_COLOUR_MAP[track_frame.object_name]
                )

                # draw track history
                thickness = int(min(self._image_annotated.shape[:2]) / 500)
                previous_point = None
                for point in summary.history:
                    if point is not None:
                        cv2.circle(
                            self._image_annotated,
                            point,
                            thickness * 2,
                            ann_colour,
                            -1,
                        )
                        if previous_point is not None:
                            cv2.line(
                                self._image_annotated,
                                previous_point,
                                point,
                                ann_colour,
                                thickness,
                            )
                    previous_point = point

                # draw bounding box if current frame is valid
                if summary.latest_detection_index == summary.frame_count - 1:

                    # draw bounding box
                    thickness = int(min(self._image_annotated.shape[:2]) / 250)
                    cv2.rectangle(
                        self._image_annotated, (x1, y1), (x2, y2), ann_colour, thickness
                    )

                    # extract object/cat label, confidence, and text size
                    label = f"{summary.track_id} {summary.cat_name or track_frame.object_name}"
                    (w, h), _ = cv2.getTextSize(label, FONT, 1, 1)

                    # draw background rectangle for text
                    txt_box_coords = (int(x1 + 1.1 * w), int(y1 + 1.2 * h))
                    cv2.rectangle(
                        self._image_annotated, (x1, y1), txt_box_coords, ann_colour, -1
                    )

                    # add text
                    txt_coords = (int(x1 + w * 0.05), int(y1 + h * 1.1))
                    cv2.putText(
                        self._image_annotated,
                        label,
                        txt_coords,
                        FONT,
                        1,
                        (255, 255, 255),
                    )

            # log annotation duration
            utils.log_timing(logger, "Image annotation", start, self.hash)

        return self._image_annotated


def _release_writers(
    wtr: Optional[FFmpegWriter],
    wtr_r: Optional[FFmpegWriter],
    frame_hash: Optional[str] = None,
    log_msg: str = "",
) -> tuple[None, None]:
    """Release video writers if not None, with optional logging."""
    hash_msg = f"({frame_hash}) " if frame_hash else ""
    log_msg = f" ({log_msg})" if log_msg else ""
    if wtr is not None:
        wtr.release()
        path_msg = f": {wtr.output_path}" if wtr.output_path else ""
        logger.warning(f"{hash_msg}Saving recording{log_msg}{path_msg}")
    if wtr_r is not None:
        wtr_r.release()
        path_msg = f": {wtr_r.output_path}" if wtr_r.output_path else ""
        logger.warning(f"{hash_msg}Saving raw recording{log_msg}{path_msg}")
    return None, None


def processing_thread():
    """Process frames to detect objects and record videos"""
    logger.info("Processing thread started")

    # initialise buffers
    pre_buffer: deque[Frame] = deque(
        maxlen=int(np.ceil(settings.FPS * settings.BUFFER_DUR))
    )
    processing_buffer: deque[Frame] = deque()
    replay_buffer: deque[tuple[datetime, np.ndarray]] = deque()

    # initialise state and previous frame
    recording = False
    writer = None
    writer_raw = None
    prev_frame = None
    frames_since_detection = 0
    track_manager = TrackManager()

    while not shutdown_event.is_set() or not frame_queue.empty() or replay_buffer:

        # get frame from capture queue
        try:
            if replay_buffer:
                timestamp, image = replay_buffer.popleft()
                logger.debug("Processing replay frame")
            else:
                timestamp, image = frame_queue.get(timeout=0.1)
            start_capture = datetime.now()
            frame_captured = Frame(
                timestamp=timestamp,
                image=image,
                prev_frame=prev_frame,
                prev_track_mask=track_manager.all_tracks_mask(
                    settings.FRAME_WIDTH, settings.FRAME_HEIGHT
                ),
                forced_detection_run=frames_since_detection + 1 >= settings.FPS,
            )
        except queue.Empty:
            continue

        logger.debug(f"({frame_captured.hash}) Running processing")

        # detect objects and update tracking state
        track_frames, did_run_detection = frame_captured.detect_objects()
        track_manager.update(track_frames, frame_captured.hash)
        frame_captured.processing_track_summaries = [
            t.summary for t in track_manager.non_expired_tracks
        ]

        # add captured frame to processing buffer
        processing_buffer.append(frame_captured)

        # update non-detection counter
        if did_run_detection:
            frames_since_detection = 0
        else:
            frames_since_detection += 1

        # update current frame to be previous frame
        prev_frame = frame_captured

        # log processing rate
        elapsed_capture = utils.log_timing(
            logger, "Processing", start_capture, frame_captured.hash
        )

        # process frames in the buffer if enough frames have been captured
        start_recording = datetime.now()
        while len(processing_buffer) > int(
            np.ceil(settings.FPS) * settings.TRACK_NEW_DUR
        ):

            # extract frame and check for excluded objects
            frame_recording = processing_buffer.popleft()
            frame_recording.recording_track_summaries = [
                track_manager.get_track(t.track_id).summary
                for t in frame_recording.processing_track_summaries
            ]
            assert all(
                [
                    s.state != TrackState.NEW
                    for s in frame_recording.recording_track_summaries
                ]
            )
            current_confirmed_summaries_recording = [
                s
                for s in frame_recording.recording_track_summaries
                if s.state in [TrackState.ACTIVE, TrackState.STALE]
            ]
            has_excluded_object = any(
                s.last_valid_frame.object_name in settings.EXCLUDED_OBJECTS
                for s in current_confirmed_summaries_recording
            )

            if any(
                s.state in [TrackState.ACTIVE, TrackState.STALE]
                for s in current_confirmed_summaries_recording
            ):

                if has_excluded_object:

                    # clear buffers
                    processing_buffer.clear()
                    pre_buffer.clear()
                    track_manager = TrackManager()
                    logger.info(
                        f"({frame_recording.hash}) Clearing buffer and Tracks due to detection of excluded object"
                    )

                else:

                    # initialise recording and write pre buffer to video file
                    if not recording:

                        # init recordings
                        if settings.SAVE_RAW_VIDEO in {"no", "both"}:
                            out_path = os.path.join(
                                settings.OUTPUT_DIR,
                                f"{frame_recording.timestamp.strftime('%Y%m%d_%H%M%S')}.avi",
                            )
                            writer = FFmpegWriter(
                                out_path,
                                settings.FPS,
                                settings.FRAME_WIDTH,
                                settings.FRAME_HEIGHT,
                                settings.MJPEG_QV,
                            )
                            logger.warning(
                                f"({frame_recording.hash}) Starting recording: {out_path}"
                            )
                        if settings.SAVE_RAW_VIDEO in {"only", "both"}:
                            out_raw_path = os.path.join(
                                settings.OUTPUT_DIR,
                                f"{frame_recording.timestamp.strftime('%Y%m%d_%H%M%S')}_raw.avi",
                            )
                            writer_raw = FFmpegWriter(
                                out_raw_path,
                                settings.FPS,
                                settings.FRAME_WIDTH,
                                settings.FRAME_HEIGHT,
                                settings.MJPEG_QV,
                            )
                            logger.warning(
                                f"({frame_recording.hash}) Starting raw recording: {out_raw_path}"
                            )

                        # flush buffer
                        pre_buffer_len = len(pre_buffer)
                        for bf in pre_buffer:
                            _ = bf.image_annotated
                        start_buf = datetime.now()
                        while pre_buffer:
                            bf = pre_buffer.popleft()
                            logger.debug(
                                "(%s) Pre-buffer write marker: remaining_pre=%s",
                                frame_recording.hash,
                                len(pre_buffer),
                            )
                            if writer is not None:
                                writer.write(bf.image_annotated)
                            if writer_raw is not None:
                                writer_raw.write(bf.image)
                        logger.info(
                            f"({frame_recording.hash}) Written {pre_buffer_len} frames from pre detection buffer"
                        )
                        utils.log_timing(
                            logger, "Buffer writing", start_buf, frame_recording.hash
                        )

                        recording = True

            if recording:

                # write current frame and assess post buffer termination
                if not has_excluded_object:
                    _ = frame_recording.image_annotated
                    start_write = datetime.now()
                    if writer is not None:
                        writer.write(frame_recording.image_annotated)
                    if writer_raw is not None:
                        writer_raw.write(frame_recording.image)
                    utils.log_timing(
                        logger,
                        "Current frame writing",
                        start_write,
                        frame_recording.hash,
                    )

                # stop recording close video file
                if not track_manager.non_expired_tracks or has_excluded_object:
                    if has_excluded_object:
                        log_msg = "excluded object detected"
                    elif not track_manager.non_expired_tracks:
                        log_msg = "all tracks expired"
                    writer, writer_raw = _release_writers(
                        writer, writer_raw, frame_recording.hash, log_msg
                    )

                    if not track_manager.non_expired_tracks:
                        replayed = len(processing_buffer)
                        if replayed:
                            while processing_buffer:
                                frame_replay = processing_buffer.popleft()
                                replay_buffer.append(
                                    (frame_replay.timestamp, frame_replay.image.copy())
                                )
                            logger.info(
                                f"({frame_recording.hash}) Queued {replayed} delayed frame(s) for replay"
                            )
                            pre_buffer.clear()
                            track_manager = TrackManager()
                            prev_frame = None
                            frames_since_detection = 0
                            logger.info(
                                f"({frame_recording.hash}) Reset processing state before replaying delayed frames"
                            )
                    recording = False

            else:

                # store current frame image and timestamp to rolling buffer
                if not has_excluded_object:
                    pre_buffer.append(frame_recording)

        # log recording rate
        elapsed_recording = utils.log_timing(
            logger, "Recording", start_recording, frame_captured.hash
        )

        # log overall FPS
        if (overall_fps := 1 / (elapsed_capture + elapsed_recording)) < settings.FPS:
            logger.warning(f"Processing/recording thread slow: {overall_fps:.1f} FPS")

    # cleanup
    if processing_buffer:
        logger.info(f"Discarding {len(processing_buffer)} delayed frame(s)")
        processing_buffer.clear()
    writer, writer_raw = _release_writers(writer, writer_raw)

    logger.info("Processing thread stopped")
