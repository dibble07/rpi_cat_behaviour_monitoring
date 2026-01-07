import logging
from datetime import datetime
from typing import List

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


class Frame:
    """Store frame image and timestamp as well as supplementary processing and annotations"""

    def __init__(self, image: np.ndarray) -> None:
        self.time = datetime.now()
        self.image = np.ascontiguousarray(image)

    @property
    def object_detections(self) -> List[dict]:
        # run object detection if not previously run
        if not hasattr(self, "_object_detections"):
            logger.info("Running object detection")

            # initialise detectiosn output
            self._object_detections = []

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

            # log detections
            if self._object_detections:
                logger.info(
                    f"Object(s) detected: {set([x["class"] for x in self._object_detections])}"
                )

        return self._object_detections

    @property
    def image_annotated(self) -> np.ndarray:
        # annotate image if not already done
        if not hasattr(self, "_image_annotated"):
            logger.debug("Annotating image")

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

        return self._image_annotated


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
