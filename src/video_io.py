import logging
import os
import queue
import random
import subprocess
import threading
from collections.abc import Callable
from datetime import datetime

import cv2
import numpy as np

import utils
from config import SYSTEM, settings

logger = logging.getLogger(__name__)


class Cv2Camera:
    def __init__(self, video_path: str | None = None) -> None:
        # initialise camera object
        self.video_path = (
            random.choice(utils.get_video_paths(mock_inputs=True))
            if video_path is None
            else video_path
        )
        logger.info(f"Using Cv2Camera with mock video: {self.video_path}")
        self.cam = cv2.VideoCapture(self.video_path)

        # check resolution set correctly
        width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._resize = width != settings.FRAME_WIDTH or height != settings.FRAME_HEIGHT
        if self._resize:
            logger.warning(
                f"Camera resolution ({width} x {height}) does not match target ({settings.FRAME_WIDTH}x{settings.FRAME_HEIGHT})"
            )

        logger.info("Camera object initialised")

    def __call__(self) -> np.ndarray | None:
        # capture frame from camera
        success, frame = self.cam.read()
        if not success:
            return None

        # resize to settings specified resolution
        if self._resize and frame is not None:
            frame = cv2.resize(frame, (settings.FRAME_WIDTH, settings.FRAME_HEIGHT))

        # store for potential buffering
        self._buffer_count = 0
        self._last_frame = frame

        logger.debug(f"Frame is of type {type(frame)}")
        return frame


class Picamera2Camera:
    def __init__(self) -> None:
        # import raspberry pi specific library
        from picamera2 import Picamera2

        # initialise camera
        self.cam = Picamera2()

        # configure camera
        config = self.cam.create_video_configuration(
            main={
                "size": (settings.FRAME_WIDTH, settings.FRAME_HEIGHT),
                "format": "RGB888",
            },
            controls={
                "FrameRate": settings.FPS,
                "AeEnable": True,
                "AeMeteringMode": 2,
                "AwbEnable": True,
                "HdrMode": 4,
            },
        )
        self.cam.configure(config)

        # start camera
        self.cam.start()

        logger.info("Camera object initialised")

    def __call__(self) -> np.ndarray:
        frame = self.cam.capture_array()[..., :3]
        logger.debug(f"Frame is of type {type(frame)} and shape {frame.shape}")
        return frame


def get_camera(
    mock_video_path: str | None = None,
) -> Cv2Camera | Picamera2Camera:
    """Create the camera source suited to the current platform"""
    match SYSTEM:
        case "Darwin":
            return Cv2Camera(mock_video_path)
        case "Linux":
            return Picamera2Camera()
        case _:
            raise ValueError(f"Unexpected system: {SYSTEM}")


class FfmpegWriter:
    """Drop-in replacement for cv2.VideoWriter using ffmpeg H.264 encoding"""

    def __init__(
        self,
        path: str | os.PathLike[str],
        fps: float,
        width: int,
        height: int,
        quality: int,
        queue_size_callback: Callable | None = None,
    ) -> None:

        # prepare output path and recording queue
        self.output_path = os.fspath(path)
        self.init_timestamp = os.path.basename(self.output_path).removesuffix(
            ".tmp.mp4"
        )
        self._queue_size_callback = queue_size_callback
        self._queue: queue.Queue = queue.Queue(maxsize=25)
        self._report_queue_size()
        self._check_queue_empty()

        # start ffmpeg encoding raw BGR frames from stdin
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

        # write queued frames in a background thread
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def _check_queue_empty(self) -> None:
        """Warn if frames are still waiting to be written"""
        pending = self._queue.qsize()
        if pending:
            logger.warning(
                f"Recording queue not empty: {pending} item(s) for {self.output_path}"
            )

    def _report_queue_size(self) -> None:
        """Share the current queue size with the optional callback"""
        if self._queue_size_callback is not None:
            self._queue_size_callback(self._queue.qsize())

    def _writer_loop(self) -> None:
        """Pipe queued frames to ffmpeg until the end-of-stream marker arrives"""

        # get next frame or stop at the end-of-stream marker
        while True:
            item = self._queue.get()
            self._report_queue_size()
            if item is None:
                break

            # write frame to ffmpeg and log timings
            enqueue_ts, frame_hash, frame = item
            utils.log_timing(logger, "FFmpeg queue delay", enqueue_ts, frame_hash)
            start = datetime.now()
            self._proc.stdin.write(frame.tobytes())
            utils.log_timing(logger, "FFmpeg write", start, frame_hash)

    def write(self, frame: np.ndarray, frame_hash: str) -> None:
        """Queue a frame for encoding"""
        start = datetime.now()
        self._queue.put((start, frame_hash, frame))
        self._report_queue_size()
        utils.log_timing(logger, "FFmpeg enqueue", start, frame_hash)

    def release(self) -> None:
        """Flush queued frames and close the ffmpeg process"""
        self._queue.put(None)
        self._report_queue_size()
        self._thread.join()
        self._proc.stdin.close()
        self._proc.wait()
        self._report_queue_size()
        self._check_queue_empty()
