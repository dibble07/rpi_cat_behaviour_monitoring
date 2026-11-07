import faulthandler
import logging

from config import APP_LOG_PATH, configure_logging

configure_logging(APP_LOG_PATH)

logger = logging.getLogger(__name__)
faulthandler.enable()

import threading

import shared
from annotate import annotate_thread
from capture import capture_thread
from monitoring import monitoring_thread
from processing import processing_thread

# start threads
capture_t = threading.Thread(target=capture_thread)
processing_t = threading.Thread(target=processing_thread)
annotation_t = threading.Thread(target=annotate_thread)
monitoring_t = threading.Thread(target=monitoring_thread)
capture_t.start()
processing_t.start()
annotation_t.start()
monitoring_t.start()

# keep main thread alive until shutdown is requested
shared.wait_for_shutdown()

# close down application
logger.info("Waiting for threads to finish...")
shared.join_thread(capture_t, "Capture")
shared.join_thread(processing_t, "Processing")
while annotation_t.is_alive():
    if not processing_t.is_alive():
        shared.processing_busy_event.clear()
        shared.processing_stopped_event.set()
    logger.info("Waiting for pending video annotations to finish...")
    annotation_t.join(timeout=5)
shared.join_thread(monitoring_t, "Monitoring")
if shared.thread_exception is not None:
    logger.error("Application shutdown due to an unhandled thread exception")
    raise SystemExit(1)
logger.info("Application shutdown complete")
