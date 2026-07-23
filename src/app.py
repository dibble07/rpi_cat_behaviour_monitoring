import faulthandler
import logging
import os
from datetime import datetime

from config import settings

# prepare output directory
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(
                settings.OUTPUT_DIR,
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_logs.txt",
            ),
            "a",
        ),
    ],
)

logger = logging.getLogger(__name__)
faulthandler.enable()

import threading
import time

from capture import capture_thread
from monitoring import monitoring_thread
from processing import processing_thread
from shared import shutdown_event

# start threads
capture_t = threading.Thread(target=capture_thread)
processing_t = threading.Thread(target=processing_thread)
monitoring_t = threading.Thread(target=monitoring_thread)
capture_t.start()
processing_t.start()
monitoring_t.start()

# keep main thread alive until shutdown is requested
try:
    while not shutdown_event.is_set():
        time.sleep(0.1)
except KeyboardInterrupt:
    shutdown_event.set()
finally:
    shutdown_event.set()

# close down application
logger.info("Waiting for threads to finish...")
capture_t.join(timeout=5)
processing_t.join(timeout=5)
monitoring_t.join(timeout=5)
logger.info("Application shutdown complete")
