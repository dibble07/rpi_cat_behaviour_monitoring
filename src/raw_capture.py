import faulthandler
import logging
import os
import queue
import threading
import time

from capture import capture_thread
from config import (
    OUTPUT_DIR,
    RAW_LOG_PATH,
    TIMESTAMP_FORMAT,
    configure_logging,
    settings,
)
from detection import Frame
from monitoring import monitoring_thread
from processing import release_writer
from shared import (
    frame_queue,
    join_thread,
    set_recording_queue_size,
    shutdown_event,
    wait_for_shutdown,
)
from video_io import FfmpegWriter

logger = logging.getLogger(__name__)

_ROTATION_SECONDS = 60 * 60


def raw_recording_thread() -> None:
    """Write raw clips from captured frames and rotate periodically"""
    logger.info("Raw recording thread started")

    # initialise writer state and rotation deadline
    writer = None
    prev_frame = None
    rotation_deadline = time.monotonic() + _ROTATION_SECONDS
    while not shutdown_event.is_set() or not frame_queue.empty():

        # get frame from capture queue
        try:
            timestamp, image = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        # hash frame for the video hash map
        frame = Frame(
            timestamp=timestamp,
            image=image,
            prev_frame=prev_frame,
            prev_track_mask=image[:, :, 0] * 0,
            forced_detection_run=False,
        )
        prev_frame = frame

        # rotate to a new video file when due
        now_mono = time.monotonic()
        if writer is None or now_mono >= rotation_deadline:
            if writer is not None:
                writer = release_writer(writer, log_msg="raw test rotation")
            writer = FfmpegWriter(
                path=os.path.join(
                    OUTPUT_DIR,
                    f"{timestamp.strftime(TIMESTAMP_FORMAT)}.tmp.mp4",
                ),
                fps=settings.FPS,
                width=settings.FRAME_WIDTH,
                height=settings.FRAME_HEIGHT,
                quality=settings.VIDEO_QUALITY,
                queue_size_callback=set_recording_queue_size,
            )
            rotation_deadline = now_mono + _ROTATION_SECONDS
            logger.warning(f"Starting raw test recording: {writer.output_path}")

        # write frame to current video
        writer.write(frame.image, frame.hash)

    # close the final video
    if writer is not None:
        writer = release_writer(writer, log_msg="raw test shutdown")
    logger.info("Raw recording thread stopped")


configure_logging(RAW_LOG_PATH)
faulthandler.enable()

# start threads
capture_t = threading.Thread(target=capture_thread)
recording_t = threading.Thread(target=raw_recording_thread)
monitoring_t = threading.Thread(target=monitoring_thread)
capture_t.start()
recording_t.start()
monitoring_t.start()

# keep main thread alive until shutdown is requested
wait_for_shutdown()

# close down application
logger.info("Waiting for threads to finish...")
join_thread(capture_t, "Capture")
join_thread(recording_t, "Raw recording")
join_thread(monitoring_t, "Monitoring")
logger.info("Application shutdown complete")
