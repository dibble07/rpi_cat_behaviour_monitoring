import logging
import os
from typing import Union

import cv2

from config import SYSTEM, settings

logger = logging.getLogger(__name__)


class Cv2_camera:
    def __init__(self):
        # initialise camera object
        video_path = os.path.join("sample_images", "20260619_131505_raw.avi")
        logger.info("Using Cv2_camera with mock video")
        self.cam = cv2.VideoCapture(video_path)

        # check resolution set correctly
        width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._resize = width != settings.FRAME_WIDTH or height != settings.FRAME_HEIGHT
        if self._resize:
            logger.warning(
                f"Camera resolution ({width} x {height}) does not match target ({settings.FRAME_WIDTH}x{settings.FRAME_HEIGHT})"
            )

        # buffer variables for final frame when video ends
        self._buffer_frames = int(settings.BUFFER_DUR * settings.FPS)
        self._last_frame = None
        self._buffer_count = 0

        logger.info("Camera object initialised")

    def __call__(self):
        # capture frame from camera
        success, frame = self.cam.read()

        # handle video end with buffering
        if not success:
            if self._buffer_count < self._buffer_frames:
                self._buffer_count += 1
                logger.info(
                    f"Buffering last frame ({self._buffer_count}/{self._buffer_frames})"
                )
                return self._last_frame
            else:
                logger.info("Buffer period expired, returning None")
                return None

        # resize to settings specified resolution
        if self._resize and frame is not None:
            frame = cv2.resize(frame, (settings.FRAME_WIDTH, settings.FRAME_HEIGHT))

        # store for potential buffering
        self._buffer_count = 0
        self._last_frame = frame

        logger.debug(f"Frame is of type {type(frame)}")
        return frame


class Picamera2_camera:
    def __init__(self):
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

    def __call__(self):
        frame = self.cam.capture_array()[..., :3]
        logger.debug(f"Frame is of type {type(frame)} and shape {frame.shape}")
        return frame


def get_camera() -> Union[Cv2_camera, Picamera2_camera]:
    match SYSTEM:
        case "Darwin":
            return Cv2_camera()
        case "Linux":
            return Picamera2_camera()
        case _:
            raise ValueError(f"Unexpected system: {SYSTEM}")
