import faulthandler
import logging
import queue
import threading
import time

from capture import capture_thread
from config import RAW_LOG_PATH, TIMESTAMP_FORMAT, settings
from ffmpegwriter import FfmpegWriter
from monitoring import monitoring_thread
from processing import Frame, _release_writers
from shared import frame_queue, shutdown_event

logger = logging.getLogger(__name__)

_ROTATION_SECONDS = 60 * 60


def raw_recording_thread() -> None:
    """Write raw clips from captured frames and rotate periodically"""
    logger.info("Raw recording thread started")

    writer: FfmpegWriter | None = None
    prev_frame: Frame | None = None
    rotation_deadline = time.monotonic() + _ROTATION_SECONDS
    while not shutdown_event.is_set() or not frame_queue.empty():
        try:
            timestamp, image = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        frame = Frame(
            timestamp=timestamp,
            image=image,
            prev_frame=prev_frame,
            prev_track_mask=image[:, :, 0] * 0,
            forced_detection_run=False,
        )
        prev_frame = frame

        now_mono = time.monotonic()
        if writer is None or now_mono >= rotation_deadline:
            if writer is not None:
                writer = _release_writers(writer, log_msg="raw test rotation")
            writer = FfmpegWriter(
                init_timestamp=timestamp.strftime(TIMESTAMP_FORMAT),
                fps=settings.FPS,
                width=settings.FRAME_WIDTH,
                height=settings.FRAME_HEIGHT,
                quality=settings.VIDEO_QUALITY,
            )
            rotation_deadline = now_mono + _ROTATION_SECONDS
            logger.warning(f"Starting raw test recording: {writer.output_path}")

        writer.write(frame.image, frame.hash)

    if writer is not None:
        writer = _release_writers(writer, log_msg="raw test shutdown")
    logger.info("Raw recording thread stopped")


logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RAW_LOG_PATH, "a"),
    ],
)

faulthandler.enable()

capture_t = threading.Thread(target=capture_thread)
recording_t = threading.Thread(target=raw_recording_thread)
monitoring_t = threading.Thread(target=monitoring_thread)
capture_t.start()
recording_t.start()
monitoring_t.start()

try:
    while not shutdown_event.is_set():
        time.sleep(0.1)
except KeyboardInterrupt:
    shutdown_event.set()
finally:
    shutdown_event.set()

logger.info("Waiting for threads to finish...")
capture_t.join(timeout=5)
if capture_t.is_alive():
    logger.warning("Capture thread did not exit cleanly within timeout")
recording_t.join(timeout=5)
if recording_t.is_alive():
    logger.warning("Raw recording thread did not exit cleanly within timeout")
monitoring_t.join(timeout=5)
if monitoring_t.is_alive():
    logger.warning("Monitoring thread did not exit cleanly within timeout")
logger.info("Application shutdown complete")
