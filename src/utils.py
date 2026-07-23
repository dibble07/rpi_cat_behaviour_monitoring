import logging
from collections import defaultdict
from datetime import datetime
from typing import Optional, Tuple

import torch


def log_timing(
    logger: logging.Logger, task: str, start_time: datetime, frame_hash: str = ""
) -> float:
    """Log task duration in milliseconds with optional frame hash context."""
    elapsed_sec = (datetime.now() - start_time).total_seconds()
    frame_hash_str = f"({frame_hash}) " if frame_hash else ""
    logger.debug(f"{frame_hash_str}{task} duration: {elapsed_sec * 1000:.1f} ms")
    return elapsed_sec


# Map annotation colours based on object name
OBJECT_COLOUR_MAP = defaultdict(
    lambda: (255, 0, 0),
    {
        "person": (0, 0, 255),
        "cat": (0, 192, 0),
    },
)


class Bbox:
    """Bounding box with lazy xyxy/xywhn conversion."""

    def __init__(
        self,
        xyxy: Optional[Tuple[int, int, int, int]] = None,
        xywhn: Optional[Tuple[float, float, float, float]] = None,
        frame_wh: Optional[Tuple[int, int]] = None,
    ) -> None:
        if (xyxy is None) == (xywhn is None):
            raise ValueError("Provide exactly one of xyxy or xywhn")

        if frame_wh is not None:
            self._frame_width, self._frame_height = frame_wh
        else:
            self._frame_width = self._frame_height = None
        self._xyxy = xyxy
        self._xywhn = xywhn

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        if self._xyxy is None:
            if self._frame_width is None or self._frame_height is None:
                raise ValueError("frame_wh is required to convert xywhn to xyxy")

            max_x = self._frame_width - 1
            max_y = self._frame_height - 1
            xc, yc, bw, bh = self._xywhn

            self._xyxy = (
                int(round((xc - bw / 2) * max_x)),
                int(round((yc - bh / 2) * max_y)),
                int(round((xc + bw / 2) * max_x)),
                int(round((yc + bh / 2) * max_y)),
            )

        return self._xyxy

    @property
    def xywhn(self) -> tuple[float, float, float, float]:
        if self._xywhn is None:
            if self._frame_width is None or self._frame_height is None:
                raise ValueError("frame_wh is required to convert xyxy to xywhn")

            max_x = self._frame_width - 1
            max_y = self._frame_height - 1
            x1, y1, x2, y2 = self._xyxy

            xc = ((x1 + x2) / 2) / max_x
            yc = ((y1 + y2) / 2) / max_y
            bw = (x2 - x1) / max_x
            bh = (y2 - y1) / max_y

            self._xywhn = (xc, yc, bw, bh)

        return self._xywhn

    @property
    def cxcywh(self) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.xyxy
        return int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2)), x2 - x1, y2 - y1


def get_best_device() -> torch.device:
    """Identify the best available PyTorch device"""
    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        out = torch.device("cuda")

    # Check for Mac GPU (Metal Performance Shaders)
    elif torch.backends.mps.is_available():
        out = torch.device("mps")

    # Fallback to CPU
    else:
        out = torch.device("cpu")

    return out


def expand_bbox_from_bounds(
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    image_width: int,
    image_height: int,
    pad: float,
    target_aspect_ratio: Optional[float] = None,
) -> list[int]:
    """Expand a bbox with padding and enforce frame aspect ratio."""

    # identify initial padded bounding box
    pad = int(pad * max(x_max - x_min, y_max - y_min))
    y1, y2 = max(0, y_min - pad), min(image_height - 1, y_max + pad)
    x1, x2 = max(0, x_min - pad), min(image_width - 1, x_max + pad)

    # calculate current and target aspect ratio
    box_h = y2 - y1 + 1
    box_w = x2 - x1 + 1
    target_ar = (
        target_aspect_ratio
        if target_aspect_ratio is not None
        else image_width / image_height
    )
    box_ar = box_w / box_h

    # calculate extra pixels needed and space either side
    if box_ar != target_ar:
        if box_ar < target_ar:
            new_w = int(round(box_h * target_ar))
            delta = new_w - box_w
            space_bef, space_aft = x1, image_width - x2 - 1
        elif box_ar > target_ar:
            new_h = int(round(box_w / target_ar))
            delta = new_h - box_h
            space_bef, space_aft = y1, image_height - y2 - 1
        else:
            raise ValueError(f"Cannot handle aspect ratios: {box_ar}, {target_ar}")

        # calculate growth either side, targetting symmetry but guaranteeing aspect ratio
        if space_bef <= space_aft:
            grow_bef = min(delta // 2, space_bef)
            grow_aft = delta - grow_bef
        else:
            grow_aft = min(delta // 2, space_aft)
            grow_bef = delta - grow_aft

        # update bounding box locations
        if box_ar < target_ar:
            x1 -= grow_bef
            x2 += grow_aft
        else:
            y1 -= grow_bef
            y2 += grow_aft

    # check aspect ratio is within rounding range
    low_ar = (x2 - x1 + 0.5) / (y2 - y1 + 1.5)
    high_ar = (x2 - x1 + 1.5) / (y2 - y1 + 0.5)
    assert low_ar <= target_ar <= high_ar

    return [int(x1), int(y1), int(x2), int(y2)]
