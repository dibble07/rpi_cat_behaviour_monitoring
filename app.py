import faulthandler
import logging
import os
import queue
import signal
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import psutil

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
process = psutil.Process(os.getpid())
faulthandler.enable()


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

    while not shutdown_event.is_set():

        # get image and timestamp
        timestamp = datetime.now()
        image = cam()

        # shutdown if mock camera reached end of file
        if cam._mock and image is None:
            logger.warning("Mock camera reached end of file")
            shutdown_event.set()
            continue

        # enqueue the frame
        frame_queue.put((timestamp, image))

        # maintain camera frame rate
        elapsed = (datetime.now() - timestamp).total_seconds()
        delay = frame_period - elapsed
        logger.debug(f"Capture duration: {elapsed*1000:.1f} ms")
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
    pre_buffer = utils.PreBuffer(max_duration=settings.BUFFER_DUR)

    # initialise state and previous frame
    recording = False
    writer = None
    prev_image_grey_blur = np.zeros(
        (settings.FRAME_HEIGHT, settings.FRAME_WIDTH), dtype=np.uint8
    )
    prev_object_detections = []  # type: ignore

    while not shutdown_event.is_set() or not frame_queue.empty():

        # get frame from capture queue
        try:
            timestamp, image = frame_queue.get(timeout=0.1)
            frame = utils.Frame(
                timestamp=timestamp,
                image=image,
                prev_image_grey_blur=prev_image_grey_blur,
                prev_object_detections=prev_object_detections,
            )
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
        prev_image_grey_blur = frame.prev_image_grey_blur.copy()
        prev_object_detections = frame.prev_object_detections.copy()

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


def monitoring_thread():
    # prepare percent sampling
    psutil.cpu_percent(percpu=True)
    process.cpu_percent()

    while not shutdown_event.is_set():
        # memory usage
        rss = process.memory_info().rss / (1024 * 1024)
        logger.info(f"Memory: {rss:.0f} MB")

        # CPU usage
        proc_cpu = process.cpu_percent(interval=None)
        logger.info(f"ProcCPU: {proc_cpu:.1f}%")

        # thread counts
        num_threads = process.num_threads()
        logger.info(f"Threads:{num_threads}")

        # queue size
        q_len = frame_queue.qsize()
        logger.info(f"Frame queue length: {q_len}")

        time.sleep(0.5)


# start threads
capture_t = threading.Thread(target=capture_thread)
processing_t = threading.Thread(target=processing_thread)
monitoring_t = threading.Thread(target=monitoring_thread)
capture_t.start()
processing_t.start()
monitoring_t.start()

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
monitoring_t.join(timeout=5)
cv2.destroyAllWindows()
logger.info("Application shutdown complete")
