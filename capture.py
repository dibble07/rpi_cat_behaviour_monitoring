import logging
import os
import time
from datetime import datetime

from config import SYSTEM
from shared import cam, frame_queue, shutdown_event

logger = logging.getLogger(__name__)


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
