import argparse
import logging
import queue
import signal
import sys
import threading
import time
import traceback
from types import FrameType

from video_io import get_camera

logger = logging.getLogger(__name__)
thread_exception = None


# prepare camera, optionally from a mock video path passed on the command line
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--video", "--video_path", dest="video_path")
_args, _ = _parser.parse_known_args(sys.argv[1:])
cam = get_camera(_args.video_path)


def _handle_exit(signum: int, _: FrameType | None) -> None:
    """Set global shutdown request flag"""
    logger.info(f"Received signal {signum} to shut down")
    shutdown_event.set()


def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
    """Trigger application shutdown when any thread raises an unhandled exception"""
    global thread_exception
    if args.exc_type in (SystemExit, KeyboardInterrupt):
        return
    thread_exception = args.exc_value
    trace = "".join(
        traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
    )
    logger.critical(f"Unhandled exception in thread '{args.thread.name}':\n{trace}")
    shutdown_event.set()


# prepare threadsafe queues and queue-size telemetry
frame_queue = queue.Queue()  # type: ignore[var-annotated]
recording_queue_size = 0
processing_busy_event = threading.Event()
processing_stopped_event = threading.Event()


def set_recording_queue_size(size: int) -> None:
    """Publish the latest recording queue size for monitoring"""
    global recording_queue_size
    recording_queue_size = size


# prepare terminal shutdown
shutdown_event = threading.Event()
signal.signal(signal.SIGINT, _handle_exit)
signal.signal(signal.SIGTERM, _handle_exit)
threading.excepthook = _thread_excepthook


def wait_for_shutdown() -> None:
    """Keep the main thread alive until shutdown is requested"""
    while not shutdown_event.is_set():
        time.sleep(0.1)


def join_thread(thread: threading.Thread, name: str) -> None:
    """Wait for a thread to finish and warn if it does not exit cleanly"""
    thread.join(timeout=5)
    if thread.is_alive():
        logger.warning(f"{name} thread did not exit cleanly within timeout")
