import cv2

from config import SYSTEM, settings


class Cv2_camera:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FPS, settings.TARGET_FPS)
        self.fps = self.cam.get(cv2.CAP_PROP_FPS)

    def __call__(self):
        _, frame = self.cam.read()
        return frame


def get_camera():
    match SYSTEM:
        case "Darwin":
            out = Cv2_camera()
        case _:
            raise ValueError(f"Unexpected system: {SYSTEM}")

    return out
