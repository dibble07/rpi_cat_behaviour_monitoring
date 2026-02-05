from __future__ import annotations

import faulthandler
import logging
import os
import queue
import threading
import time
from datetime import datetime

import cv2

from capture import capture_thread
from config import SYSTEM, settings
from monitoring import monitoring_thread
from processing import processing_thread
from shared import display_queue, shutdown_event

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
faulthandler.enable()

# prepare output directory
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

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
