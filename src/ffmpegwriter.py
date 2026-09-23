import logging
import os
import queue
import subprocess
import threading
from collections.abc import Callable
from datetime import datetime

import numpy as np

import utils
from config import OUTPUT_DIR

logger = logging.getLogger(__name__)


class FfmpegWriter:
    """Drop-in replacement for cv2.VideoWriter using ffmpeg H.264 encoding."""

    def __init__(
        self,
        init_timestamp: str,
        fps: float,
        width: int,
        height: int,
        quality: int,
        queue_size_callback: Callable[[int], None] | None = None,
    ) -> None:
        self.init_timestamp = init_timestamp
        self._queue_size_callback = queue_size_callback
        filename = f"{self.init_timestamp}.tmp.mp4"
        self.output_path = os.path.join(OUTPUT_DIR, filename)
        self._queue: queue.Queue = queue.Queue(maxsize=25)
        self._report_queue_size()
        self._check_queue_empty()
        cmd = [
            "ffmpeg",
            "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(quality),
            "-movflags",
            "faststart",
            "-pix_fmt",
            "yuv420p",
            self.output_path,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if self._proc.poll() is not None:
            raise RuntimeError(f"ffmpeg failed to start for {self.output_path}")
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def _check_queue_empty(self) -> None:
        pending = self._queue.qsize()
        if pending:
            logger.warning(
                f"Recording queue not empty: {pending} item(s) for {self.output_path}"
            )

    def _report_queue_size(self) -> None:
        if self._queue_size_callback is not None:
            self._queue_size_callback(self._queue.qsize())

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            self._report_queue_size()
            if item is None:
                break

            enqueue_ts, frame_hash, frame = item
            utils.log_timing(logger, "FFmpeg queue delay", enqueue_ts, frame_hash)
            start = datetime.now()
            self._proc.stdin.write(frame.tobytes())
            utils.log_timing(logger, "FFmpeg write", start, frame_hash)

    def write(self, frame: np.ndarray, frame_hash: str) -> None:
        start = datetime.now()
        self._queue.put((start, frame_hash, frame))
        self._report_queue_size()
        utils.log_timing(logger, "FFmpeg enqueue", start, frame_hash)

    def release(self) -> None:
        self._queue.put(None)
        self._report_queue_size()
        self._thread.join()
        self._proc.stdin.close()
        self._proc.wait()
        self._report_queue_size()
        self._check_queue_empty()
