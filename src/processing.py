from __future__ import annotations

import hashlib
import logging
import os
import queue
import subprocess
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import List, Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO

import utils
from config import SYSTEM, settings
from shared import cam, display_queue, frame_queue, shutdown_event
from tracking import TrackFrame, TrackManager, TrackState, TrackSummary

logger = logging.getLogger(__name__)

# annotation font
FONT = cv2.FONT_HERSHEY_SIMPLEX

# load object detection model
MODEL = YOLO(settings.MODEL_PATH, task="detect")
_ = MODEL(
    np.zeros((settings.FRAME_HEIGHT, settings.FRAME_WIDTH, 3), dtype=np.uint8),
    imgsz=settings.IMGSZ,
    conf=settings.CONF,
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
        cmd = [
            "ffmpeg",
            "-y",
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
            "-maxrate",
            "5M",
            "-bufsize",
            "6M",
            path,
        ]
        logger.warning(f"FFmpegWriter cmd: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self._proc.poll() is not None:
            raise RuntimeError(f"ffmpeg failed to start for {path}")
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def _writer_loop(self) -> None:
        while True:
            frame = self._queue.get()
            if frame is None:
                break
            self._proc.stdin.write(frame.tobytes())

    def write(self, frame: np.ndarray) -> None:
        self._queue.put(frame)

    def release(self) -> None:
        self._queue.put(None)
        self._thread.join()
        self._proc.stdin.close()
        rc = self._proc.wait()
        stderr_bytes = self._proc.stderr.read()
        if rc != 0:
            logger.error(
                f"FFmpegWriter: ffmpeg failed (returncode={rc}):\n"
                f"{stderr_bytes.decode(errors='replace')[-500:]}"
            )


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
        grey_blur_elapsed = (datetime.now() - start).total_seconds()
        start = datetime.now()
        self.hash = hashlib.md5(self.image_grey_blur.tobytes()).hexdigest()[:6]
        elapsed = (datetime.now() - start).total_seconds()
        logger.debug(f"({self.hash}) Blur duration: {grey_blur_elapsed*1000:.1f} ms")
        logger.debug(f"({self.hash}) Hash duration: {elapsed*1000:.1f} ms")

        if prev_frame is None:
            logger.warning(f"No previous frame provided")
            self.prev_image_grey_blur = np.zeros((_GREY_H, _GREY_W), dtype=np.uint8)
        else:
            self.prev_image_grey_blur = prev_frame.image_grey_blur.copy().astype(
                np.uint8
            )

    def _detect_motion(self):

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
        prev_mask = cv2.resize(
            self.prev_track_mask,
            (_GREY_W, _GREY_H),
            interpolation=cv2.INTER_NEAREST,
        )

        # combine motion and previous detection masks into the next detection region
        search_mask = cv2.bitwise_or(motion_mask, prev_mask)

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
        elapsed = (datetime.now() - start).total_seconds()
        logger.debug(f"({self.hash}) Motion detection duration: {elapsed*1000:.1f} ms")

        # log motion
        if self._has_search_area:
            logger.info(f"({self.hash}) Has search area: {self._has_search_area}")

    @property
    def search_mask(self) -> np.ndarray:
        if not hasattr(self, "_search_mask"):
            self._detect_motion()
        return self._search_mask

    @property
    def has_search_area(self) -> bool:
        if not hasattr(self, "_has_search_area"):
            self._detect_motion()
        return self._has_search_area

    def _identify_search_bbox(self):

        logger.debug(f"({self.hash}) Identifying search bounding box")

        # identify bbox of the next detection region
        h, w = self.image.shape[:2]
        ys, xs = np.where(self.search_mask > 0)
        x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
        self._search_bbox = utils.expand_bbox_from_bounds(
            x_min, x_max, y_min, y_max, w, h, int(0.1 * h)
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
                logger.info(f"({self.hash}) forced object detection run")

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
                imgsz=settings.IMGSZ,
                conf=settings.CONF,
                verbose=False,
                max_det=settings.MAX_DETS,
            )[0]

            # process detections
            detected_classes = set()
            for r in results.boxes:
                bbox = tuple(r.xyxy[0].cpu().numpy().astype(np.int32) + offsets)
                class_id = int(r.cls[0].item())
                track_frames.append(
                    TrackFrame(
                        frame_hash=self.hash,
                        bbox=utils.Bbox(xyxy=bbox),
                        class_id=class_id,
                        confidence=float(r.conf[0].item()),
                    )
                )
                detected_classes.add(class_id)

            # log detection duration
            elapsed = (datetime.now() - start).total_seconds()
            logger.debug(
                f"({self.hash}) Object detection duration: {elapsed*1000:.1f} ms"
            )

            # log detections
            if detected_classes:
                logger.info(f"({self.hash}) Object(s) detected: {detected_classes}")
        else:
            did_run_detection = False
        return track_frames, did_run_detection

    @property
    def track_summaries(self) -> List[TrackSummary]:
        if not hasattr(self, "_track_summaries"):
            raise RuntimeError(f"Track summaries not set yet for frame {self.hash}")
        return self._track_summaries

    @track_summaries.setter
    def track_summaries(self, track_summaries: List[TrackSummary]) -> None:
        self._track_summaries = track_summaries

    @property
    def image_annotated(self) -> np.ndarray:
        if not hasattr(self, "_image_annotated"):
            if not hasattr(self, "_track_summaries"):
                raise RuntimeError(f"Track summaries not set yet for frame {self.hash}")

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
            for summary in self._track_summaries:
                track_frame = summary.last_valid_frame

                # unpack box coords
                x1, y1, x2, y2 = track_frame.bbox.xyxy
                ann_colour = utils.CLASS_COLOUR_MAP[track_frame.class_id]

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

                    # extract object class label/confidence and text size
                    label = f"{MODEL.names[track_frame.class_id]} {summary.track_id} - {summary.state.name[0]}{summary.frame_count-summary.first_detection_index} {track_frame.confidence*100:.0f}%"
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
            elapsed = (datetime.now() - start).total_seconds()
            logger.debug(
                f"({self.hash}) Image annotation duration: {elapsed*1000:.1f} ms"
            )

        return self._image_annotated


class PreBuffer:
    def __init__(self, max_duration: Union[int, float]):
        self.max_duration = max_duration
        self.frames: List[Frame] = []

    def __iter__(self):
        return self

    def __next__(self):
        if not self.frames:
            raise StopIteration
        return self.frames.pop(0)

    def __len__(self):
        return len(self.frames)

    def _sort(self):
        self.frames.sort(key=lambda x: x.timestamp)

    def check_duration(self, time):
        min_time = time - timedelta(seconds=settings.BUFFER_DUR)
        self.frames = [x for x in self.frames if x.timestamp >= min_time]
        self._sort()

    def put(self, frame: Frame):
        self.frames.append(frame)
        self.check_duration(frame.timestamp)


def processing_thread():
    """Process frames to detect objects and record videos"""
    logger.info("Processing thread started")

    # initialise buffers
    pre_buffer = PreBuffer(max_duration=settings.BUFFER_DUR)
    processing_buffer: deque[Frame] = deque()

    # initialise state and previous frame
    recording = False
    writer = None
    writer_raw = None
    prev_frame = None
    frames_since_detection = 0
    track_manager = TrackManager()

    while not shutdown_event.is_set() or not frame_queue.empty():

        # get frame from capture queue
        try:
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
        start_track = datetime.now()
        track_manager.update(track_frames)
        frame_captured.track_summaries = [
            t.summary for t in track_manager.non_expired_tracks
        ]
        track_elapsed = (datetime.now() - start_track).total_seconds()
        logger.debug(
            f"({frame_captured.hash}) Tracking duration: {track_elapsed*1000:.1f} ms"
        )

        # add captured frame to processing buffer
        processing_buffer.append(frame_captured)

        # send frame to display queue
        if SYSTEM == "Darwin":
            try:
                display_queue.put_nowait(frame_captured.image_annotated.copy())
            except queue.Full:
                pass

        # update non-detection counter
        if did_run_detection:
            frames_since_detection = 0
        else:
            frames_since_detection += 1

        # update current frame to be previous frame
        prev_frame = frame_captured

        # log processing rate
        elapsed_capture = (datetime.now() - start_capture).total_seconds()
        logger.debug(
            f"({frame_captured.hash}) Processing duration: {elapsed_capture*1000:.1f} ms"
        )

        # process frames in the buffer if enough frames have been captured
        start_recording = datetime.now()
        while len(processing_buffer) > int(np.ceil(settings.FPS)):

            # extract frame and check for excluded classes
            frame_recording = processing_buffer.popleft()
            track_id_recording = [t.track_id for t in frame_recording.track_summaries]
            current_summaries_recording = [
                track_manager.get_track(id).summary for id in track_id_recording
            ]
            assert all([s.confirmed is not None for s in current_summaries_recording])
            current_confirmed_summaries_recording = [
                s for s in current_summaries_recording if s.confirmed
            ]
            has_excluded_class = any(
                s.last_valid_frame.class_id in settings.EXCLUDED_CLASSES
                for s in current_confirmed_summaries_recording
            )

            if any(
                s.state == TrackState.ACTIVE
                for s in current_confirmed_summaries_recording
            ):

                if has_excluded_class:

                    # clear buffers
                    processing_buffer.clear()
                    pre_buffer.frames.clear()
                    track_manager = TrackManager()
                    logger.info("Clearing buffer due to detection of excluded class")

                else:

                    # initialise recording and write pre buffer to video file
                    if not recording:

                        # init recordings
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
                        logger.warning(f"Starting recording: {out_path}")
                        if settings.SAVE_RAW_VIDEO:
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

                        # flush buffer
                        pre_buffer_len = len(pre_buffer)
                        start_buf = datetime.now()
                        for bf in pre_buffer:
                            writer.write(bf.image_annotated)
                            if settings.SAVE_RAW_VIDEO:
                                writer_raw.write(bf.image)
                        logger.info(
                            f"Written {pre_buffer_len} frames from pre detection buffer"
                        )
                        logger.debug(
                            f"({frame_recording.hash}) Buffer writing duration: {(datetime.now() - start_buf).total_seconds()*1000:.1f} ms"
                        )

                        recording = True

            if recording:

                # write current frame and assess post buffer termination
                if not has_excluded_class:
                    start_write = datetime.now()
                    writer.write(frame_recording.image_annotated)
                    if settings.SAVE_RAW_VIDEO:
                        writer_raw.write(frame_recording.image)
                    logger.debug(
                        f"({frame_recording.hash}) Current frame writing duration: {(datetime.now() - start_write).total_seconds()*1000:.1f} ms"
                    )

                # stop recording close video file
                if not track_manager.non_expired_tracks or has_excluded_class:
                    writer.release()
                    writer = None
                    if settings.SAVE_RAW_VIDEO:
                        writer_raw.release()
                        writer_raw = None
                    if not track_manager.non_expired_tracks:
                        logger.info("Saving clip: all tracks expired")
                    elif has_excluded_class:
                        logger.info(f"Saving clip: excluded class detected")
                    recording = False

            else:

                # store current frame image and timestamp to rolling buffer
                if not has_excluded_class:
                    pre_buffer.put(frame_recording)

        # log recording rate
        elapsed_recording = (datetime.now() - start_recording).total_seconds()
        logger.debug(
            f"({frame_captured.hash}) Recording duration: {elapsed_recording*1000:.1f} ms"
        )

        # log overall FPS
        if (overall_fps := 1 / (elapsed_capture + elapsed_recording)) < settings.FPS:
            logger.warning(f"Processing thread slow: {overall_fps:.1f} FPS")

    # cleanup
    if processing_buffer:
        logger.info(f"Discarding {len(processing_buffer)} delayed frame(s)")
        processing_buffer.clear()
    if writer is not None:
        writer.release()
        logger.info("Saving clip")
    if writer_raw is not None:
        writer_raw.release()
        logger.info("Saving raw clip")

    logger.info("Processing thread stopped")
