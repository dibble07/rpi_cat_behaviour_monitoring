from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import settings

logger = logging.getLogger(__name__)


ANN_COLOUR = (0, 200, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# load object detection model
MODEL = YOLO(settings.MODEL_PATH, task="detect")

# define background subtractor
BACK_SUB = cv2.createBackgroundSubtractorMOG2(history=100, detectShadows=False)


class Frame:
    """Store frame image and timestamp as well as supplementary processing and annotations"""

    def __init__(self, image: np.ndarray, prev_frame: Optional[Frame]) -> None:
        self.time = datetime.now()
        self.image = np.ascontiguousarray(image)
        self.hash = hashlib.md5(image.tobytes()).hexdigest()[:6]
        self.prev_frame = prev_frame

    @property
    def image_grey_blur(self) -> np.ndarray:
        if not hasattr(self, "_image_grey_blur"):
            grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self._image_grey_blur = cv2.GaussianBlur(grey, (5, 5), 0)
        return self._image_grey_blur

    def _detect_motion(self):

        if not self.prev_frame:
            logger.warning(
                f"({self.hash}) No previous frame available for motion detection"
            )
            # assume no motion if frame has no previous
            self._motion_mask = np.zeros_like(self.image)
            self._has_motion = False
        else:
            # start timing
            start = datetime.now()
            logger.debug(f"({self.hash}) Running motion detection")

            # calculate mask of changes from the previous frame
            diff = cv2.absdiff(self.prev_frame.image_grey_blur, self.image_grey_blur)
            _, diff_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

            # get mask of foreground from background removal model
            fore_mask = BACK_SUB.apply(self.image)

            # combine change and foreground masks
            mask = cv2.bitwise_or(diff_mask, fore_mask)

            # smooth regions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.dilate(mask, kernel)

            # store motion mask and presence flag
            self._motion_mask = mask
            self._has_motion = mask.mean() / 255 > 0.01

            # log detection duration
            elapsed = (datetime.now() - start).total_seconds()
            logger.debug(
                f"({self.hash}) Motion detection duration: {elapsed*1000:.1f} ms"
            )

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
        if self.has_motion or (
            self.prev_frame is not None and self.prev_frame.object_detections
        ):
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

            # log detections
            if self._object_detections:
                logger.info(
                    f"({self.hash}) Object(s) detected: {set([x["class"] for x in self._object_detections])}"
                )

    @property
    def object_detections(self) -> List[dict]:
        # run object detection if not previously run
        if not hasattr(self, "_object_detections"):
            self._detect_objects()
        return self._object_detections

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
        self.frames.sort(key=lambda x: x.time)

    def check_duration(self, time):
        min_time = time - timedelta(seconds=settings.BUFFER_DUR)
        self.frames = [x for x in self.frames if x.time >= min_time]
        self._sort()

    def put(self, frame: Frame):
        self.frames.append(frame)
        self.check_duration(frame.time)


def get_best_device() -> torch.device:
    """Identify the best available PyTorch device"""
    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        out = torch.device("cuda")

    # Check for Mac GPU (Metal Performance Shaders)
    elif torch.backends.mps.is_available():
        out = torch.device("mps")

    # Fallback to CPU
    else:
        out = torch.device("cpu")

    return out
