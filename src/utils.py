import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import psutil

logger = logging.getLogger(__name__)


def get_rss_mb() -> float:
    """Return current process RSS memory in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def log_import_memory(label: str, before_mb: float) -> float:
    """Log RSS memory before/after/delta for a significant import block.

    Usage::

        _m = get_rss_mb()
        import heavy_package
        log_import_memory("heavy_package", _m)

    Returns the current RSS in MB so it can be chained.
    """
    after_mb = get_rss_mb()
    delta_mb = after_mb - before_mb
    logger.info(
        "[mem] %-30s  before=%6.1f MB  after=%6.1f MB  delta=%+.1f MB",
        label,
        before_mb,
        after_mb,
        delta_mb,
    )
    return after_mb


def get_video_paths(mock_inputs: bool = True, raw_video: bool = True) -> list[Path]:
    """Return video paths from selected dataset sources"""

    # identify source directories
    datasets_root = Path("datasets")
    source_dirs = []
    if mock_inputs:
        source_dirs.append(datasets_root / "mock_inputs")
    if raw_video:
        source_dirs.append(datasets_root / "raw_video")
    if not source_dirs:
        raise FileNotFoundError(f"No source directories found for selected sources")

    # identify all paths with matching extensions
    exts = ("*.avi", "*.mp4")
    video_paths = [
        path
        for source_dir in source_dirs
        for ext in exts
        for path in source_dir.glob(ext)
    ]
    if not video_paths:
        raise FileNotFoundError(
            f"No paths found in selected sources with extension(s): {exts}"
        )

    return sorted(video_paths)


def log_timing(
    logger: logging.Logger,
    task: str,
    start_time: datetime,
    frame_hash: str = "",
    level: int = logging.DEBUG,
) -> float:
    """Log task duration in milliseconds with optional frame hash and log level."""
    elapsed_sec = (datetime.now() - start_time).total_seconds()
    frame_hash_str = f"({frame_hash}) " if frame_hash else ""
    logger.log(level, f"{frame_hash_str}{task} duration: {elapsed_sec * 1000:.1f} ms")
    return elapsed_sec


# Map annotation colours based on object or cat name
OBJECT_COLOUR_MAP = {
    "person": (0, 0, 255),  # red
    "cat": (0, 192, 0),  # green
}
CAT_COLOUR_MAP = {
    "fluffy": (255, 0, 0),  # blue
    "tabby": (0, 165, 255),  # orange
    "na": (19, 69, 139),  # brown
}


class Bbox:
    """Bounding box with lazy conversion between formats"""

    def __init__(
        self,
        xyxy: Optional[Tuple[int, int, int, int]] = None,
        cxcywhn: Optional[Tuple[float, float, float, float]] = None,
        frame_wh: Optional[Tuple[int, int]] = None,
    ) -> None:
        if (xyxy is None) == (cxcywhn is None):
            raise ValueError("Provide exactly one of xyxy or cxcywhn")

        if frame_wh is not None:
            self._frame_width, self._frame_height = frame_wh
        else:
            self._frame_width = self._frame_height = None
        self._xyxy = xyxy
        self._cxcywhn = cxcywhn

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """Pixel-space corner coordinates"""
        if self._xyxy is None:
            if self._frame_width is None or self._frame_height is None:
                raise ValueError("frame_wh is required to convert cxcywhn to xyxy")

            max_x = self._frame_width - 1
            max_y = self._frame_height - 1
            xc, yc, bw, bh = self._cxcywhn

            self._xyxy = (
                int(round((xc - bw / 2) * max_x)),
                int(round((yc - bh / 2) * max_y)),
                int(round((xc + bw / 2) * max_x)),
                int(round((yc + bh / 2) * max_y)),
            )

        return self._xyxy

    @property
    def cxcywhn(self) -> tuple[float, float, float, float]:
        """Normalized centroid and width coordinates"""
        if self._cxcywhn is None:
            if self._frame_width is None or self._frame_height is None:
                raise ValueError("frame_wh is required to convert xyxy to cxcywhn")

            max_x = self._frame_width - 1
            max_y = self._frame_height - 1
            x1, y1, x2, y2 = self._xyxy

            xc = ((x1 + x2) / 2) / max_x
            yc = ((y1 + y2) / 2) / max_y
            bw = (x2 - x1) / max_x
            bh = (y2 - y1) / max_y

            self._cxcywhn = (xc, yc, bw, bh)

        return self._cxcywhn

    @property
    def cxcywh(self) -> tuple[int, int, int, int]:
        """Pixel-space centroid and width coordinates"""
        x1, y1, x2, y2 = self.xyxy
        return int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2)), x2 - x1, y2 - y1


def get_best_device():
    """Identify the best available PyTorch device"""
    import torch  # notebooks only — save memory in prod

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

    # clip outputs to image bounds
    x1, x2 = int(max(0, x1)), int(min(image_width - 1, x2))
    y1, y2 = int(max(0, y1)), int(min(image_height - 1, y2))

    # check aspect ratio is within rounding range
    exact_ar = (x2 - x1) / (y2 - y1)
    low_ar = (x2 - x1 + 0.5) / (y2 - y1 + 1.5)
    high_ar = (x2 - x1 + 1.5) / (y2 - y1 + 0.5)
    if not (low_ar <= target_ar <= high_ar):
        logger.warning(
            f"Expanded bbox aspect ratio {exact_ar:.3f} is not close to target target {target_ar:.3f}"
        )

    return [x1, y1, x2, y2]


def entropy_weights(probs: np.ndarray) -> np.ndarray:
    """Calculate weights based on probability entropy"""
    ent = -np.sum(np.clip(probs, 1e-12, 1) * np.log(np.clip(probs, 1e-12, 1)), axis=1)
    return np.clip(1.0 - (ent / np.log(probs.shape[1])), 0, 1) ** 2
