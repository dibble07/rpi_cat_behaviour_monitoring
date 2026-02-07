from __future__ import annotations

import hashlib
import logging
import os
import queue
from datetime import datetime, timedelta
from typing import List, Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO

from config import SYSTEM, settings
from shared import cam, display_queue, frame_queue, shutdown_event

logger = logging.getLogger(__name__)

# annotation values
ANN_COLOUR = (0, 200, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# video type
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore

# load object detection model
MODEL = YOLO(settings.MODEL_PATH, task="detect")

# define background subtractor
BACK_SUB = cv2.createBackgroundSubtractorMOG2(
    history=settings.BACKGROUND_HISTORY, detectShadows=False
)


class Frame:
    """Store frame image and timestamp as well as supplementary processing and annotations"""

    def __init__(
        self,
        timestamp: datetime,
        image: np.ndarray,
        prev_frame: Optional[Frame],
    ) -> None:
        self.timestamp = timestamp
        self.image = np.ascontiguousarray(image)
        self.hash = hashlib.md5(image.tobytes()).hexdigest()[:6]

        if prev_frame is None:
            logger.warning(f"No previous frame provided")
            self.prev_image_grey_blur = np.zeros(
                (settings.FRAME_HEIGHT, settings.FRAME_WIDTH), dtype=np.uint8
            )
            self.prev_object_detections = []  # type: ignore
        else:
            self.prev_image_grey_blur = prev_frame.image_grey_blur.copy()
            self.prev_object_detections = prev_frame.object_detections.copy()

    @property
    def image_grey_blur(self) -> np.ndarray:
        if not hasattr(self, "_image_grey_blur"):
            grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self._image_grey_blur = cv2.GaussianBlur(grey, (5, 5), 0)
        return self._image_grey_blur

    def _detect_motion(self):

        # start timing
        start = datetime.now()
        logger.debug(f"({self.hash}) Running motion detection")

        # calculate mask of changes from the previous frame
        diff = cv2.absdiff(self.prev_image_grey_blur, self.image_grey_blur)
        _, diff_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # get mask of foreground from background removal model
        fore_mask = BACK_SUB.apply(self.image)

        # combine change and foreground masks
        motion_mask = cv2.bitwise_or(diff_mask, fore_mask)

        # smooth regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.dilate(motion_mask, kernel)

        # get mask of previous detections
        prev_mask = np.zeros_like(self.image_grey_blur)
        for o in self.prev_object_detections:
            prev_mask[o["box"][1] : o["box"][3], o["box"][0] : o["box"][2]] = 255

        # combine motion and previous detection masks
        mask = cv2.bitwise_or(motion_mask, prev_mask)

        # store motion mask and presence flag
        self._motion_mask = mask
        self._has_motion = mask.mean() / 255 > 0.001

        # log detection duration
        elapsed = (datetime.now() - start).total_seconds()
        logger.debug(f"({self.hash}) Motion detection duration: {elapsed*1000:.1f} ms")

        # log motion
        if self._has_motion:
            logger.info(f"({self.hash}) Motion detected: {self._has_motion}")

    @property
    def motion_mask(self) -> np.ndarray:
        if not hasattr(self, "_motion_mask"):
            self._detect_motion()
        return self._motion_mask

    @property
    def has_motion(self) -> bool:
        if not hasattr(self, "_has_motion"):
            self._detect_motion()
        return self._has_motion

    def _detect_objects(self):
        # initialise detections output
        self._object_detections = []

        # only run detection if motion is present
        if self.has_motion or self.prev_object_detections:
            # start timing
            start = datetime.now()
            logger.debug(f"({self.hash}) Running object detection")

            # run model inference
            results = MODEL(
                self.image,
                imgsz=settings.IMGSZ,
                verbose=False,
                max_det=settings.MAX_DETS,
            )[0]

            # process detections
            for r in results.boxes:
                self._object_detections.append(
                    {
                        "box": r.xyxy[0].detach().cpu().int().tolist(),
                        "conf": float(r.conf[0].item()),
                        "class": int(r.cls[0].item()),
                    }
                )

            # log detection duration
            elapsed = (datetime.now() - start).total_seconds()
            logger.debug(
                f"({self.hash}) Object detection duration: {elapsed*1000:.1f} ms"
            )

            # identify excluded classes
            self._has_excluded_class = bool(
                {x["class"] for x in self._object_detections}.intersection(
                    settings.EXCLUDED_CLASSES
                )
            )

            # log detections
            if self._object_detections:
                logger.info(
                    f"({self.hash}) Object(s) detected: {set([x["class"] for x in self._object_detections])}"
                )

        else:
            self._has_excluded_class = False

    @property
    def object_detections(self) -> List[dict]:
        # run object detection if not previously run
        if not hasattr(self, "_object_detections"):
            self._detect_objects()
        return self._object_detections

    @property
    def has_excluded_class(self) -> bool:
        # run object detection if not previously run
        if not hasattr(self, "_has_excluded_class"):
            self._detect_objects()
        return self._has_excluded_class

    @property
    def image_annotated(self) -> np.ndarray:
        # annotate image if not already done
        if not hasattr(self, "_image_annotated"):
            # start timing
            start = datetime.now()
            logger.debug(f"({self.hash}) Annotating image")

            # copy image ready to be annotated
            self._image_annotated = self.image.copy()

            # loop over all detections
            for obj in self.object_detections:

                # unpack box coords
                x1, y1, x2, y2 = obj["box"]

                # draw bounding box
                thickness = int(min(self._image_annotated.shape[:2]) / 250)
                cv2.rectangle(
                    self._image_annotated, (x1, y1), (x2, y2), ANN_COLOUR, thickness
                )

                # extract object class label/confidence and text size
                label = f"{MODEL.names[obj["class"]]} {obj["conf"]:.2f}"
                (w, h), _ = cv2.getTextSize(label, FONT, 1, 1)

                # draw background rectangle for text
                txt_box_coords = (int(x1 + 1.1 * w), int(y1 + 1.2 * h))
                cv2.rectangle(
                    self._image_annotated, (x1, y1), txt_box_coords, ANN_COLOUR, -1
                )

                # add text
                txt_coords = (int(x1 + w * 0.05), int(y1 + h * 1.1))
                cv2.putText(
                    self._image_annotated, label, txt_coords, FONT, 1, (255, 255, 255)
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

    # initialise preroll buffer
    pre_buffer = PreBuffer(max_duration=settings.BUFFER_DUR)

    # initialise state and previous frame
    recording = False
    writer = None
    prev_frame = None

    while not shutdown_event.is_set() or not frame_queue.empty():

        # get frame from capture queue
        try:
            timestamp, image = frame_queue.get(timeout=0.1)
            frame = Frame(timestamp=timestamp, image=image, prev_frame=prev_frame)
        except queue.Empty:
            continue

        # start timing
        logger.debug(f"({frame.hash}) Running processing")
        start = datetime.now()

        if frame.object_detections:

            if frame.has_excluded_class:

                # clear buffer
                pre_buffer.frames.clear()
                logger.info("Clearing buffer due to detection of excluded class")

            else:

                # update latest detection timestamp
                last_detection_time = frame.timestamp

                # initialise recording and write pre buffer to video file
                if not recording:
                    out_path = os.path.join(
                        settings.OUTPUT_DIR,
                        f"{frame.timestamp.strftime('%Y%m%d_%H%M%S')}.mp4",
                    )
                    writer = cv2.VideoWriter(
                        out_path, FOURCC, cam.fps, frame.image.shape[:2][::-1]
                    )
                    logger.warning(f"Starting recording: {out_path}")
                    pre_buffer_len = len(pre_buffer)
                    for bf in pre_buffer:
                        writer.write(bf.image_annotated)
                    logger.info(
                        f"Written {pre_buffer_len} frames from pre detection buffer"
                    )
                    recording = True

        if recording:

            # write current frame and assess post buffer termination
            if not frame.has_excluded_class:
                writer.write(frame.image_annotated)
                last_detection_dur = (
                    frame.timestamp - last_detection_time
                ).total_seconds()

            # stop recording close video file
            if (last_detection_dur > settings.BUFFER_DUR) or frame.has_excluded_class:
                writer.release()
                if last_detection_dur > settings.BUFFER_DUR:
                    logger.info(
                        f"Saving clip: last detection was {last_detection_dur:.3f} ago"
                    )
                elif frame.has_excluded_class:
                    logger.info(f"Saving clip: excluded class detected")
                recording = False

        else:

            # store current frame image and timestamp to rolling buffer
            if not frame.has_excluded_class:
                pre_buffer.put(frame)

        # send frame to display queue
        if SYSTEM == "Darwin":
            try:
                display_queue.put_nowait(frame.image_annotated.copy())
            except queue.Full:
                pass

        # update current frame to be previous frame
        prev_frame = frame

        # log processing rate
        elapsed = (datetime.now() - start).total_seconds()
        processing_fps = 1 / elapsed
        logger.debug(f"({frame.hash}) Processing duration: {elapsed*1000:.1f} ms")
        if processing_fps < cam.fps:
            logger.warning(f"Processing thread slow: {processing_fps:.1f} FPS")

    # cleanup
    if writer is not None:
        writer.release()
        logger.info("Saving clip")

    logger.info("Processing thread stopped")
