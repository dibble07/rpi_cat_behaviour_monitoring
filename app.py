import collections
import logging
import os
import time
from datetime import datetime, timedelta

import cv2
import numpy as np
from ultralytics import YOLO

from config import settings

logger = logging.getLogger(__name__)

# TO DO
# - Add motion detector pre check
# - Handle misaligned frame and processing timings
# - Understand timing bottlenecks

# set constants
ANN_COLOUR = (0, 200, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore


# add annotations to image
def draw_detections(
    frame: np.ndarray,
    boxes: list,
    confs: list,
    classes: list,
    names: dict,
):
    for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, classes):
        label = f"{names[int(cls)]} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, FONT, 1, 1)
        txt_coords = (int(x1 + w * 0.05), int(y1 + h * 1.1))
        txt_box_coords = (int(x1 + 1.1 * w), int(y1 + 1.2 * h))
        cv2.rectangle(frame, (x1, y1), (x2, y2), ANN_COLOUR, 2)
        cv2.rectangle(frame, (x1, y1), txt_box_coords, ANN_COLOUR, -1)
        cv2.putText(frame, label, txt_coords, FONT, 1, (255, 255, 255))


# prepare output directory
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

# prepare model
model = YOLO(settings.MODEL_PATH, task="detect")

# prepare camera and buffer
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, settings.TARGET_FPS)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps != settings.TARGET_FPS:
    logger.warning(f"Camera FPS ({fps}) does not match target ({settings.TARGET_FPS})")
frame_int = 1 / fps
pre_buffer = collections.deque(maxlen=int(settings.BUFFER_DUR * fps))  # type: ignore

# initialise state
recording = False
last_detection_time = datetime.now()
writer = None

# loop indefinitely
while True:

    # store current frame image and timestamp to rolling buffer
    t0 = datetime.now()
    _, frame = cap.read()
    pre_buffer.append((t0, frame.copy()))

    # identify objects in current frame
    results = model(
        frame, imgsz=settings.IMGSZ, verbose=False, max_det=settings.MAX_DETS
    )[0]
    boxes, confs, classes = [], [], []
    for r in results.boxes:
        boxes.append(r.xyxy[0].numpy().astype(int))
        confs.append(r.conf[0])
        classes.append(r.cls[0])
    object_present = bool(boxes)

    # start video writing and write buffer to file
    if object_present:
        last_detection_time = t0
        if not recording:
            out_path = os.path.join(
                settings.OUTPUT_DIR, f"{t0.strftime('%Y%m%d_%H%M%S')}.mp4"
            )
            writer = cv2.VideoWriter(out_path, FOURCC, fps, frame.shape[:2][::-1])
            for _, bf in pre_buffer:
                writer.write(bf)
            recording = True
            logger.warning(f"Started recording: {out_path}")

    # annotate current frame and write to file then close recording
    if recording:
        if object_present:
            draw_detections(frame, boxes, confs, classes, names=model.names)
        writer.write(frame)
        if t0 - last_detection_time > timedelta(seconds=settings.BUFFER_DUR):
            writer.release()
            logger.warning(f"Saved clip: {out_path}")
            recording = False

    # manual closing of app and recording
    cv2.imshow("Object monitor (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        if writer is not None:
            writer.release()
            logger.warning(f"Saved clip: {out_path}")
        break

    # delay next processing to match camera frame rate
    delay = (timedelta(seconds=frame_int) - (datetime.now() - t0)).total_seconds()
    if delay > 0:
        logger.warning(f"Delay next processing by {delay:.3f} seconds")
        time.sleep(delay)
