import logging
import queue
import signal
import threading
import time
import traceback
from datetime import datetime

import numpy as np

from camera import get_camera

logger = logging.getLogger(__name__)


# prepare camera
cam = get_camera()


def _handle_exit(signum: int, _: object) -> None:
    """Set global shutdown request flag"""
    set_shutdown_cause(f"signal-{signum}", force=True)
    logger.info(f"Received signal {signum} to shut down")
    shutdown_event.set()


def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
    """Trigger application shutdown when any thread raises an unhandled exception"""
    if args.exc_type in (SystemExit, KeyboardInterrupt):
        return
    set_shutdown_cause(f"thread-exception:{args.thread.name}", force=True)
    logger.critical(
        f"Unhandled exception in thread '{args.thread.name}':\n"
        + "".join(
            traceback.format_exception(
                args.exc_type, args.exc_value, args.exc_traceback
            )
        )
    )
    shutdown_event.set()


# prepare threadsafe queues
frame_queue: queue.Queue[tuple[datetime, np.ndarray]] = queue.Queue()

# shared heartbeat diagnostics
heartbeat_lock = threading.Lock()
processing_heartbeat_monotonic = 0.0
writer_heartbeat_monotonic = 0.0
processing_last_event = "not-started"
writer_last_event = "not-started"

# shared shutdown-cause diagnostics
shutdown_cause_lock = threading.Lock()
shutdown_cause = "not-requested"


def set_shutdown_cause(cause: str, force: bool = False) -> None:
    """Set process-level shutdown cause for diagnostics."""
    global shutdown_cause
    with shutdown_cause_lock:
        if force or shutdown_cause == "not-requested":
            shutdown_cause = cause


def get_shutdown_cause() -> str:
    """Return process-level shutdown cause for diagnostics."""
    with shutdown_cause_lock:
        return shutdown_cause


def update_processing_heartbeat(event: str) -> None:
    """Update processing-thread heartbeat timestamp and latest event."""
    global processing_heartbeat_monotonic, processing_last_event
    with heartbeat_lock:
        processing_heartbeat_monotonic = time.monotonic()
        processing_last_event = event


def update_writer_heartbeat(event: str) -> None:
    """Update writer heartbeat timestamp and latest event."""
    global writer_heartbeat_monotonic, writer_last_event
    with heartbeat_lock:
        writer_heartbeat_monotonic = time.monotonic()
        writer_last_event = event


def get_heartbeat_snapshot() -> tuple[float, float, str, str]:
    """Return a snapshot of heartbeat timestamps and latest events."""
    with heartbeat_lock:
        return (
            processing_heartbeat_monotonic,
            writer_heartbeat_monotonic,
            processing_last_event,
            writer_last_event,
        )


# prepare terminal shutdown
shutdown_event = threading.Event()
signal.signal(signal.SIGINT, _handle_exit)
signal.signal(signal.SIGTERM, _handle_exit)
threading.excepthook = _thread_excepthook
