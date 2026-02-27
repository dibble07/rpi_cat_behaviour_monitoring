import logging
import os
from typing import Union

import cv2

from config import SYSTEM, settings

logger = logging.getLogger(__name__)


class Cv2_camera:
    def __init__(self):
        # check for mocking
        self._mock = os.environ.get("DEV_MOCK_CAMERA") == "1"

        # initialise camera object
        if self._mock:
            video_path = os.path.join("sample_images", "20260127_072635.mp4")
            logger.info("Using Cv2_camera with mock video")
            self.cam = cv2.VideoCapture(video_path)
        else:
            logger.info("Using Cv2_camera with live video feed from source 0")
            self.cam = cv2.VideoCapture(0)

        # set camera frame rate - controlled by capture thread as hardware is fixed
        self.fps = min(settings.FPS, self.cam.get(cv2.CAP_PROP_FPS))
        logger.info(f"Camera FPS: {self.fps}")

        # set camera resolution
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
        self.width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # check resolution set correctly
        if self.width != settings.FRAME_WIDTH or self.height != settings.FRAME_HEIGHT:
            logger.warning(
                f"Camera resolution ({self.width} x {self.height}) does not match target ({settings.FRAME_WIDTH}x{settings.FRAME_HEIGHT})"
            )
        else:
            logger.info(f"Camera resolution: {self.width} x {self.height}")

        logger.info("Camera object initialised")

    def __call__(self):
        # capture frame from camera
        _, frame = self.cam.read()

        logger.debug(f"Frame is of type {type(frame)}")
        return frame


class Picamera2_camera:
    def __init__(self):
        # import raspberry pi specific library
        from picamera2 import Picamera2

        # initialise camera
        self.cam = Picamera2()

        # configure camera
        self._mock = False
        self.fps = settings.FPS
        self.width = settings.FRAME_WIDTH
        self.height = settings.FRAME_HEIGHT
        config = self.cam.create_video_configuration(
            main={
                "size": (settings.FRAME_WIDTH, settings.FRAME_HEIGHT),
                "format": "RGB888",
            },
            controls={
                "FrameRate": self.fps,
                "AeEnable": True,
                "ExposureTime": settings.EXPOSURETIME,
                "AnalogueGain": 0,
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
