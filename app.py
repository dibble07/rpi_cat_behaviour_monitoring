import collections
import logging
import os
import queue
import signal
import threading
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
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
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
# - Avoid current frame adding to buffer and after detection


def _handle_exit(signum, _):
    """Store global shutdown request flag"""
    logger.info(f"Received signal {signum} to shut down")
    shutdown_event.set()


# set constants
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore

# prep terminal shutdown
shutdown_event = threading.Event()
signal.signal(signal.SIGINT, _handle_exit)
signal.signal(signal.SIGTERM, _handle_exit)

# prepare output directory
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

# prepare model
model = YOLO(settings.MODEL_PATH, task="detect")

# prepare threadsafe queues
frame_queue: queue.Queue[tuple[datetime, np.ndarray]] = queue.Queue()
display_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)

# prepare camera
cam = get_camera()


def capture_thread():
    """Continuously capture camera frames and add to queue"""
    logger.info("Capture thread started")

    # set capture constants
    frame_period = 1 / cam.fps

    # initialise previous frame
    prev_frame = None

    while not shutdown_event.is_set():

        # capture frame and timestamp
        t0 = datetime.now()
        frame = np.ascontiguousarray(cam())

        # enqueue the frame
        frame_queue.put((t0, frame))

        # maintain camera frame rate
        elapsed = (datetime.now() - t0).total_seconds()
        delay = frame_period - elapsed
        if delay > 0:
            logger.debug(f"Capture delayed to maintain frame rate: {delay*1000:.3f} ms")
            time.sleep(delay)
        else:
            logger.warning(f"Capture thread slow: {1/elapsed:.1f} FPS")

    logger.info("Capture thread stopped")


def processing_thread():
    """Process frames to detect objects and record videos"""
    logger.info("Processing thread started")

    # initialise preroll buffer
    pre_buffer = collections.deque(maxlen=int(settings.BUFFER_DUR * cam.fps))  # type: ignore

    # initialise state
    recording = False

    while not shutdown_event.is_set() or not frame_queue.empty():

        # get frame from capture queue
        try:
            t0, frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # store current frame image and timestamp to rolling buffer
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

        if object_present:

            # update latest detection timestamp
            last_detection_time = t0

            # annotate current frame and write to video file
            utils.draw_detections(frame, boxes, confs, classes, names=model.names)

            # initialise recording and write pre buffer to video file
            if not recording:
                out_path = os.path.join(
                    settings.OUTPUT_DIR, f"{t0.strftime('%Y%m%d_%H%M%S')}.mp4"
                )
                writer = cv2.VideoWriter(
                    out_path, FOURCC, cam.fps, frame.shape[:2][::-1]
                )
                logger.warning(f"Starting recording: {out_path}")
                pre_buffer_len = len(pre_buffer)
                for _, bf in pre_buffer:
                    writer.write(bf)
                logger.info(
                    f"Written {pre_buffer_len} frames from pre detection buffer"
                )
                recording = True

        if recording:

            # write current frame and assess post buffer termination
            writer.write(frame)
            last_detection_dur = (t0 - last_detection_time).total_seconds()

            # stop recording close video file
            if last_detection_dur > settings.BUFFER_DUR:
                writer.release()
                logger.info(
                    f"Saving clip: last detection was {last_detection_dur:.3f} ago"
                )
                recording = False

        # send frame to display queue
        if SYSTEM == "Darwin":
            try:
                display_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

    # cleanup
    if writer is not None:
        writer.release()
        logger.info("Saving clip")

    logger.info("Processing thread stopped")


# start threads
capture_t = threading.Thread(target=capture_thread)
processing_t = threading.Thread(target=processing_thread)
capture_t.start()
processing_t.start()

# display frames and check for shutdown requests
try:
    while not shutdown_event.is_set():
        if SYSTEM == "Darwin":
            try:
                display_frame = display_queue.get(timeout=0.1)
                cv2.imshow("Object monitor (q to quit)", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    shutdown_event.set()
            except queue.Empty:
                cv2.waitKey(1)
        else:
            time.sleep(0.1)
except KeyboardInterrupt:
    shutdown_event.set()

# close down application
logger.info("Waiting for threads to finish...")
capture_t.join(timeout=5)
processing_t.join(timeout=5)
cv2.destroyAllWindows()
logger.info("Application shutdown complete")
