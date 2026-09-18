import argparse
import logging
import queue
import signal
import sys
import threading
import traceback
from datetime import datetime

import numpy as np

from camera import get_camera

logger = logging.getLogger(__name__)
thread_exception: BaseException | None = None


def _get_mock_video_path_from_argv() -> str | None:
    """Extract an optional mock video path from process argv."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--video", "--video_path", dest="video_path")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.video_path


# prepare camera
cam = get_camera(_get_mock_video_path_from_argv())


def _handle_exit(signum, _):
    """Set global shutdown request flag"""
    logger.info(f"Received signal {signum} to shut down")
    shutdown_event.set()


def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
    """Trigger application shutdown when any thread raises an unhandled exception"""
    global thread_exception
    if args.exc_type in (SystemExit, KeyboardInterrupt):
        return
    thread_exception = args.exc_value
    logger.critical(
        f"Unhandled exception in thread '{args.thread.name}':\n"
        + "".join(
            traceback.format_exception(
                args.exc_type, args.exc_value, args.exc_traceback
            )
        )
    )
    shutdown_event.set()


# prepare threadsafe queues and queue-size telemetry
frame_queue: queue.Queue[tuple[datetime, np.ndarray]] = queue.Queue()
recording_queue_size = 0


def set_recording_queue_size(size: int) -> None:
    global recording_queue_size
    recording_queue_size = size


# prepare terminal shutdown
shutdown_event = threading.Event()
signal.signal(signal.SIGINT, _handle_exit)
signal.signal(signal.SIGTERM, _handle_exit)
threading.excepthook = _thread_excepthook
