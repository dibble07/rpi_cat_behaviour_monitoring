import hashlib
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def blur_image(image: np.ndarray) -> np.ndarray:
    """Create a fixed-size resized greyscale blur of an image."""
    return cv2.GaussianBlur(
        cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (640, 480)), (5, 5), 0
    )


def hash_image(image: np.ndarray) -> str:
    """Return a short content hash for an image array."""
    if image.ndim != 2:
        image = blur_image(image)
    return hashlib.md5(image.tobytes()).hexdigest()[:8]


@lru_cache(maxsize=None)
def _get_video_hashes(path: str) -> tuple[str, ...]:
    capture = cv2.VideoCapture(path)
    hashes = []
    while True:
        success, image = capture.read()
        if not success:
            break
        hashes.append(hash_image(image))
    capture.release()
    return tuple(hashes)


def get_video_hashes(video_path: str | Path) -> tuple[str, ...]:
    """Return cached per-frame hashes for every decodable frame of a video."""
    return _get_video_hashes(str(Path(video_path).resolve()))


def get_video_paths(
    mock_inputs: bool = False,
    mock_inputs_long: bool = False,
    raw_behaviour: bool = False,
    raw_detection_identification: bool = False,
) -> list[Path]:
    """Return video paths from selected dataset sources"""

    # identify source directories
    datasets_root = Path("datasets")
    source_dirs = []
    if mock_inputs:
        source_dirs.append(datasets_root / "mock_inputs")
    if mock_inputs_long:
        source_dirs.append(datasets_root / "mock_inputs_long")
    if raw_behaviour:
        source_dirs.append(datasets_root / "raw_behaviour")
    if raw_detection_identification:
        source_dirs.append(datasets_root / "raw_detection_identification")
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
    wgt = np.clip(1.0 - (ent / np.log(probs.shape[1])), 0, 1) ** 2
    return np.ones_like(wgt) if wgt.sum() == 0 else wgt


def bbox_overlap(box_a: Bbox, box_b: Bbox) -> tuple[float, float]:
    """Calculate intersection-over-union and intersection-over-minimum area"""

    # unpack box coords
    ax1, ay1, ax2, ay2 = box_a.xyxy
    bx1, by1, bx2, by2 = box_b.xyxy

    # determine intersection coord
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    # calculate areas
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = area_a + area_b - inter_area
    min_area = min(area_a, area_b)

    return inter_area / union_area, inter_area / min_area


def nms_boxes(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, iom_threshold: float
) -> list:
    """Return confidence-ranked indices after IoU/IoM NMS on one class of xyxy boxes"""
    if len(boxes) == 0:
        return []

    sizes = boxes[:, 2:] - boxes[:, :2]
    valid = (sizes > 0).all(axis=1)
    indices = np.flatnonzero(valid)
    order = indices[np.argsort(-scores[indices])]
    bboxes = [Bbox(xyxy=tuple(box)) for box in boxes]
    keep = []

    while order.size:
        selected = order[0]
        keep.append(selected)
        remaining = order[1:]
        if not remaining.size:
            break
        iou, iom = np.asarray(
            [bbox_overlap(bboxes[selected], bboxes[index]) for index in remaining]
        ).T
        order = remaining[(iou <= iou_threshold) & (iom < iom_threshold)]

    return keep
