import collections
import logging
import os
import signal
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

import utils
from camera import get_camera
from config import SYSTEM, settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(
                settings.OUTPUT_DIR,
                f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            ),
            "a",
        ),
    ],
)

logger = logging.getLogger(__name__)

# TO DO
# - Handle misaligned frame and processing timings
# - Avoid current frame adding to buffer and after detection


def _handle_exit(signum, _):
    """Store global shutdown request flag"""
    global shutdown_requested
    logger.info(f"Received signal {signum} to shut down")
    shutdown_requested = True


# set constants
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore

# prep terminal shutdown
shutdown_requested = False
signal.signal(signal.SIGINT, _handle_exit)
signal.signal(signal.SIGTERM, _handle_exit)


# prepare output directory
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

# prepare model
model = YOLO(settings.MODEL_PATH, task="detect")

# prepare camera and buffer
cam = get_camera()
frame_int = 1 / cam.fps
pre_buffer = collections.deque(maxlen=int(settings.BUFFER_DUR * cam.fps))  # type: ignore

# initialise state
recording = False
last_detection_time = datetime.now()
writer = None

# loop indefinitely
while True:

    # store current frame image and timestamp to rolling buffer
    t0 = datetime.now()
    frame = np.ascontiguousarray(cam())
    pre_buffer.append((t0, frame.copy()))

    # identify objects in current frame
    results = model(
        frame, imgsz=settings.IMGSZ, verbose=False, max_det=settings.MAX_DETS
    )[0]
    boxes, confs, classes = [], [], []
    for r in results.boxes:
        boxes.append(r.xyxy[0].detach().cpu().int().tolist())
        confs.append(float(r.conf[0].item()))
        classes.append(int(r.cls[0].item()))
    object_present = bool(boxes)
    if object_present:
        logger.info(
            f"Object(s) detected: boxes = {boxes}, confs = {confs}, classes = {classes}"
        )

    # start video writing and write buffer to file
    if object_present:
        last_detection_time = t0
        if not recording:
            out_path = os.path.join(
                settings.OUTPUT_DIR, f"{t0.strftime('%Y%m%d_%H%M%S')}.mp4"
            )
            writer = cv2.VideoWriter(out_path, FOURCC, cam.fps, frame.shape[:2][::-1])
            logger.warning(f"Starting recording: {out_path}")
            pre_buffer_len = len(pre_buffer)
            for _, bf in pre_buffer:
                writer.write(bf)
            logger.info(f"Written {pre_buffer_len} frames from pre detection buffer")
            recording = True

    # annotate current frame and write to file then close recording
    if recording:
        if object_present:
            utils.draw_detections(frame, boxes, confs, classes, names=model.names)
        writer.write(frame)
        last_detection_dur = (t0 - last_detection_time).total_seconds()
        if last_detection_dur > settings.BUFFER_DUR:
            writer.release()
            logger.info(f"Saving clip: last detection was {last_detection_dur:.3f} ago")
            recording = False

    # manual closing of app and recording
    if SYSTEM == "Darwin":
        cv2.imshow("Object monitor (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q") or shutdown_requested:
        if writer is not None:
            writer.release()
        logger.info("Saving clip: manually closed")
        break

    # delay next processing to match camera frame rate
    elapsed = (datetime.now() - t0).total_seconds()
    delay = frame_int - elapsed
    logger.info(f"Processing rate is {(1/elapsed):.1f} FPS")
    if delay > 0:
        logger.info(f"Delay next processing by {delay:.3f} seconds")
        time.sleep(delay)
