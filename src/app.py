import atexit
import faulthandler
import logging
import os
import threading
import time
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

from capture import capture_thread
from monitoring import monitoring_thread
from processing import processing_thread
from shared import get_shutdown_cause, set_shutdown_cause, shutdown_event


def _log_process_exit() -> None:
    """Emit a final process marker to distinguish clean exits from hard kills."""
    logger.info("Process exit marker reached (atexit)")


def _join_and_log(thread: threading.Thread, timeout: float = 5.0) -> None:
    """Join thread and log whether it terminated before timeout."""
    thread.join(timeout=timeout)
    if thread.is_alive():
        logger.warning(
            "Thread still alive after %.1fs join timeout: %s", timeout, thread.name
        )
    else:
        logger.info("Thread joined cleanly: %s", thread.name)


atexit.register(_log_process_exit)
logger.info(
    "Process started: pid=%s ppid=%s cwd=%s output_dir=%s",
    os.getpid(),
    os.getppid(),
    os.getcwd(),
    settings.OUTPUT_DIR,
)

# start threads
capture_t = threading.Thread(target=capture_thread, name="capture")
processing_t = threading.Thread(target=processing_thread, name="processing")
monitoring_t = threading.Thread(target=monitoring_thread, name="monitoring")
capture_t.start()
processing_t.start()
monitoring_t.start()
logger.info(
    "Threads started: capture_alive=%s processing_alive=%s monitoring_alive=%s",
    capture_t.is_alive(),
    processing_t.is_alive(),
    monitoring_t.is_alive(),
)

# keep main thread alive until shutdown is requested
try:
    while not shutdown_event.is_set():
        time.sleep(0.1)
except KeyboardInterrupt:
    set_shutdown_cause("keyboard-interrupt", force=True)
    shutdown_event.set()
finally:
    if get_shutdown_cause() == "not-requested":
        set_shutdown_cause("main-loop-exit")
    shutdown_event.set()

# close down application
logger.info("Waiting for threads to finish (cause=%s)...", get_shutdown_cause())
_join_and_log(capture_t)
_join_and_log(processing_t)
_join_and_log(monitoring_t)
logger.info("Application shutdown complete (cause=%s)", get_shutdown_cause())
