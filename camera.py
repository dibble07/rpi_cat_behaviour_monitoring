import logging
from typing import Union

import cv2

from config import SYSTEM, settings

logger = logging.getLogger(__name__)


class Cv2_camera:
    def __init__(self):
        # initialise camera obejct
        self.cam = cv2.VideoCapture(0)

        # attempt to set camera frame rate
        self.cam.set(cv2.CAP_PROP_FPS, settings.FPS)
        self.fps = self.cam.get(cv2.CAP_PROP_FPS)

        # check frame rate set correctly
        if self.fps != settings.FPS:
            logger.warning(
                f"Camera FPS ({self.fps}) does not match target ({settings.FPS})"
            )
        else:
            logger.info(f"Camera FPS: {self.fps}")

        # attempt to set camera resolution
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
        width = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # check resolution set correctly
        if width != settings.FRAME_WIDTH or height != settings.FRAME_HEIGHT:
            logger.warning(
                f"Camera resolution ({width}x{height}) does not match target ({settings.FRAME_WIDTH}x{settings.FRAME_HEIGHT})"
            )
        else:
            logger.info(f"Camera resolution: {width}x{height}")

        logger.info(f"Camera object initialised: {self.cam}")

    def __call__(self):
        _, frame = self.cam.read()
        logger.debug(f"Frame is of type {type(frame)} and shape {frame.shape}")
        return frame


class Picamera2_camera:
    def __init__(self):
        # import raspberry pi specific library
        from picamera2 import Picamera2

        # initialise camera obejct
        self.cam = Picamera2()

        # set camera resolution and frame rate
        self.fps = settings.FPS
        config = self.cam.create_video_configuration(
            main={"size": (settings.FRAME_WIDTH, settings.FRAME_HEIGHT)},
            controls={"FrameRate": self.fps},
        )
        self.cam.configure(config)

        # start camera object
        self.cam.start()

        logger.info(f"Camera object initialised: {self.cam}")

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
