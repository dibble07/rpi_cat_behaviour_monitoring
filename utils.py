import logging

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


ANN_COLOUR = (0, 200, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


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


def draw_detections(
    frame: np.ndarray,
    boxes: list,
    confs: list,
    classes: list,
    names: dict,
):
    """Annotate image with identified objects"""
    for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, classes):

        # draw bounding box
        thickness = int(min(frame.shape[:2]) / 250)
        cv2.rectangle(frame, (x1, y1), (x2, y2), ANN_COLOUR, thickness)

        # extract object class label/confidence and text size
        label = f"{names[int(cls)]} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, FONT, 1, 1)

        # draw background rectangle for text
        txt_box_coords = (int(x1 + 1.1 * w), int(y1 + 1.2 * h))
        cv2.rectangle(frame, (x1, y1), txt_box_coords, ANN_COLOUR, -1)

        # add text
        txt_coords = (int(x1 + w * 0.05), int(y1 + h * 1.1))
        cv2.putText(frame, label, txt_coords, FONT, 1, (255, 255, 255))

        logger.debug(
            f"Annotated frame with object: class = {cls}, x1y1x2y2 = {x1},{y1},{x2},{y2}"
        )
