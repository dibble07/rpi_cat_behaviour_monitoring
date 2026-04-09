import logging
import queue
import signal
import threading
import traceback
from datetime import datetime

import numpy as np

from camera import get_camera

logger = logging.getLogger(__name__)


# prepare camera
cam = get_camera()


def _handle_exit(signum, _):
    """Set global shutdown request flag"""
    logger.info(f"Received signal {signum} to shut down")
    shutdown_event.set()


def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
    """Trigger application shutdown when any thread raises an unhandled exception"""
    if args.exc_type in (SystemExit, KeyboardInterrupt):
        return
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
display_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)

# prepare terminal shutdown
shutdown_event = threading.Event()
signal.signal(signal.SIGINT, _handle_exit)
signal.signal(signal.SIGTERM, _handle_exit)
threading.excepthook = _thread_excepthook
