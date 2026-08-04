import logging
import os
import time
from datetime import datetime

import utils
from config import SYSTEM, settings
from shared import cam, frame_queue, shutdown_event

logger = logging.getLogger(__name__)


def capture_thread() -> None:
    """Continuously capture camera frames and add to queue"""
    logger.info("Capture thread started")

    # set capture constants
    frame_period = 1 / settings.FPS
    frame_period = frame_period / 4 if SYSTEM == "Darwin" else frame_period

    while not shutdown_event.is_set():

        # get image and timestamp
        timestamp = datetime.now()
        image = cam()

        # shutdown if camera source ended (can occur with video capture or device errors)
        if image is None:
            logger.warning(
                f"Camera source ended with queue size: {frame_queue.qsize()}, shutting down capture thread"
            )
            shutdown_event.set()
            continue

        # enqueue the frame
        frame_queue.put((timestamp, image))

        # maintain camera frame rate
        elapsed = utils.log_timing(logger, "Capture", timestamp)
        delay = frame_period - elapsed
        if delay > 0:
            logger.debug(f"Capture delayed to maintain frame rate: {delay*1000:.1f} ms")
            time.sleep(delay)
        else:
            logger.warning(f"Capture thread slow: {1/elapsed:.1f} FPS")

    logger.info("Capture thread stopped")
