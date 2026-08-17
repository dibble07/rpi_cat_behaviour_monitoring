import logging
import queue
import subprocess
import threading
from datetime import datetime

import numpy as np

import utils
from shared import set_recording_queue_size

logger = logging.getLogger(__name__)


class FFmpegWriter:
    """Drop-in replacement for cv2.VideoWriter using ffmpeg MPEG-4 encoding."""

    def __init__(
        self,
        path: str,
        fps: float,
        width: int,
        height: int,
        qv: int,
        raw: bool = False,
    ):
        self.output_path = path
        self._raw = raw
        self._queue: queue.Queue = queue.Queue(maxsize=25)
        set_recording_queue_size(self._queue.qsize(), raw=self._raw)
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
            "mpeg4",
            "-q:v",
            str(qv),
            "-pix_fmt",
            "yuv420p",
            path,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=None,
        )
        if self._proc.poll() is not None:
            raise RuntimeError(f"ffmpeg failed to start for {path}")
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def _check_queue_empty(self) -> None:
        pending = self._queue.qsize()
        if pending:
            logger.warning(
                f"Recording queue not empty: {pending} item(s) for {self.output_path}"
            )

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            set_recording_queue_size(self._queue.qsize(), raw=self._raw)
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
        set_recording_queue_size(self._queue.qsize(), raw=self._raw)
        utils.log_timing(logger, "FFmpeg enqueue", start, frame_hash)

    def release(self) -> None:
        self._queue.put(None)
        set_recording_queue_size(self._queue.qsize(), raw=self._raw)
        self._thread.join()
        self._proc.stdin.close()
        self._proc.wait()
        set_recording_queue_size(self._queue.qsize(), raw=self._raw)
        self._check_queue_empty()
