import logging
from typing import Union

import cv2

from config import SYSTEM, settings

logger = logging.getLogger(__name__)


class Cv2_camera:
    def __init__(self):
        # initialise camera object
        self.cam = cv2.VideoCapture(0)

        # attempt to set camera frame rate
        self.cam.set(cv2.CAP_PROP_FPS, settings.TARGET_FPS)
        self.fps = self.cam.get(cv2.CAP_PROP_FPS)

        # check frame rate set correctly
        if self.cam.fps != settings.TARGET_FPS:
            logger.warning(
                f"Camera FPS ({self.cam.fps}) does not match target ({settings.TARGET_FPS})"
            )

    def __call__(self):
        _, frame = self.cam.read()
        return frame


class Picamera2_camera:
    def __init__(self):
        # import raspberry pi specific library
        from picamera2 import Picamera2

        # initialise camera obejct
        self.cam = Picamera2()

        # find sensor mode with best resolution that meets target frame rate
        max_res_mode = max(
            [x for x in self.cam.sensor_modes if x["fps"] >= settings.TARGET_FPS],
            key=lambda m: m["size"][0] * m["size"][1],
        )

        # set camera resolution and frame rate
        self.fps = settings.TARGET_FPS
        config = self.cam.create_video_configuration(
            main={"size": max_res_mode["size"]},
            controls={"FrameRate": self.fps},
        )
        self.cam.configure(config)

        # start camera object
        self.cam.start()

    def __call__(self):
        frame = self.cam.capture_array()[..., :3]
        return frame


def get_camera() -> Union[Cv2_camera, Picamera2_camera]:
    match SYSTEM:
        case "Darwin":
            return Cv2_camera()
        case "Linux":
            return Picamera2_camera()
        case _:
            raise ValueError(f"Unexpected system: {SYSTEM}")
