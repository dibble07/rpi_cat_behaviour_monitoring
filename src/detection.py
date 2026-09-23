from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import cv2
import ncnn
import numpy as np
import yaml  # type: ignore[import-untyped]

import utils
from config import settings
from tracking import TrackFrame, TrackSummary

logger = logging.getLogger(__name__)


class _Tensor:

    def __init__(self, values: np.ndarray) -> None:
        self._arr = np.asarray(values)

    def __getitem__(self, idx: int | slice) -> _Tensor:
        return _Tensor(self._arr[idx])

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._arr

    def item(self) -> float:
        return float(self._arr.item())


class _Box:
    def __init__(
        self, x1: float, y1: float, x2: float, y2: float, conf: float, cls: float
    ) -> None:
        self.xyxy = _Tensor([[x1, y1, x2, y2]])
        self.conf = _Tensor([conf])
        self.cls = _Tensor([cls])


class YoloNcnn:

    def __init__(
        self,
        model: str | Path,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
    ) -> None:
        # load static model config and class names
        model_dir = Path(model)
        self.conf, self.iou, self.max_det = conf, iou, max_det
        metadata = yaml.safe_load((model_dir / "metadata.yaml").read_text()) or {}
        self.names = {int(k): str(v) for k, v in metadata.get("names", {}).items()}

        # load NCNN model
        self._net = ncnn.Net()
        self._net.load_param(str(model_dir / "model.ncnn.param"))
        self._net.load_model(str(model_dir / "model.ncnn.bin"))

    def predict(
        self,
        image: np.ndarray,
        imgsz: int | tuple[int, int] = 640,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: Optional[int] = None,
        verbose: bool = True,
    ) -> list[SimpleNamespace]:
        eff_conf = self.conf if conf is None else conf
        eff_iou = self.iou if iou is None else iou
        eff_max_det = self.max_det if max_det is None else max_det

        h, w = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz

        # preprocess frame colour order and scale
        src_h, src_w = image.shape[:2]
        scale_x, scale_y = src_w / w, src_h / h
        chw = np.ascontiguousarray(
            (
                cv2.resize(image, (w, h))[:, :, ::-1].astype(np.float32) / 255.0
            ).transpose(2, 0, 1)
        )

        # run NCNN inference
        with self._net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(chw).clone())
            _, out0 = ex.extract("out0")

        pred = np.asarray(out0)
        if pred.shape[0] == 6:
            pred = pred.T

        # split class logits/probabilities from box channels
        cls_scores = pred[:, 4:]
        cls_idx = cls_scores.argmax(axis=1)
        cls_conf = cls_scores.max(axis=1)

        # drop invalid/low-confidence
        valid = np.isfinite(cls_conf) & (cls_conf >= eff_conf)
        if not np.any(valid):
            return [SimpleNamespace(names=self.names, boxes=[])]
        pred = pred[valid]
        cls_idx = cls_idx[valid]
        cls_conf = cls_conf[valid]
        boxes = np.column_stack(
            (
                pred[:, 0] - pred[:, 2] / 2.0,
                pred[:, 1] - pred[:, 3] / 2.0,
                pred[:, 0] + pred[:, 2] / 2.0,
                pred[:, 1] + pred[:, 3] / 2.0,
            )
        )
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] * scale_x, 0, src_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] * scale_y, 0, src_h - 1)

        # run class-wise NMS, then merge selected indices
        keep: list[int] = []
        for cls in np.unique(cls_idx):
            idx = np.where(cls_idx == cls)[0]
            picked = cv2.dnn.NMSBoxes(
                np.column_stack(
                    (
                        boxes[idx, 0],
                        boxes[idx, 1],
                        boxes[idx, 2] - boxes[idx, 0],
                        boxes[idx, 3] - boxes[idx, 1],
                    )
                ).tolist(),
                cls_conf[idx].tolist(),
                eff_conf,
                eff_iou,
            )
            if len(picked):
                keep.extend(idx[np.asarray(picked).reshape(-1)].tolist())
        if not keep:
            return [SimpleNamespace(names=self.names, boxes=[])]

        # rank all retained boxes by confidence and clip to max_det
        keep_idx = np.asarray(keep, dtype=np.int32)
        keep_idx = keep_idx[np.argsort(-cls_conf[keep_idx])][:eff_max_det]

        return [
            SimpleNamespace(
                names=self.names,
                boxes=[
                    _Box(
                        boxes[i, 0],
                        boxes[i, 1],
                        boxes[i, 2],
                        boxes[i, 3],
                        cls_conf[i],
                        cls_idx[i],
                    )
                    for i in keep_idx
                ],
            )
        ]

    __call__ = predict


# load object detection model
_model = YoloNcnn(Path("models") / settings.MODEL_DETECTION_PATH)
_ = _model(
    np.zeros((settings.FRAME_HEIGHT, settings.FRAME_WIDTH, 3), dtype=np.uint8),
    imgsz=tuple(settings.DETECTION_IMGSZ),
    conf=settings.CONF,
    iou=settings.NMS_IOU_THRESHOLD,
    max_det=settings.MAX_DETS,
)

# define background subtractor
_back_sub = cv2.createBackgroundSubtractorMOG2(
    history=settings.BACKGROUND_HISTORY, detectShadows=False
)

# low-resolution dimensions used for motion detection
_GREY_W = 640
_GREY_H = 480


class Frame:
    """Store frame image, timestamp, and supplementary processing state"""

    def __init__(
        self,
        timestamp: datetime,
        image: np.ndarray,
        prev_frame: Optional[Frame],
        prev_track_mask: np.ndarray,
        forced_detection_run: bool,
    ) -> None:
        self.timestamp = timestamp
        self.image = np.ascontiguousarray(image)
        self.prev_track_mask = prev_track_mask
        self.forced_detection_run = forced_detection_run
        start = datetime.now()
        self.image_grey_blur = cv2.GaussianBlur(
            cv2.resize(
                cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), (_GREY_W, _GREY_H)
            ),
            (5, 5),
            0,
        )
        self.hash = hashlib.md5(self.image_grey_blur.tobytes()).hexdigest()[:6]
        utils.log_timing(logger, "Blur and hash", start, self.hash)

        if prev_frame is None:
            logger.warning(f"({self.hash}) No previous frame provided")
            self.prev_image_grey_blur = np.zeros((_GREY_H, _GREY_W), dtype=np.uint8)
        else:
            self.prev_image_grey_blur = prev_frame.image_grey_blur.copy().astype(
                np.uint8
            )

    def _identify_search_area(self):
        """Identify search area via frame differencing, background subtraction and previous tracks."""
        # start timing
        start = datetime.now()
        logger.debug(f"({self.hash}) Running motion detection")

        # calculate mask of changes from the previous frame
        diff = cv2.absdiff(self.prev_image_grey_blur, self.image_grey_blur)
        _, diff_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # get mask of foreground from background removal model
        fore_mask = _back_sub.apply(self.image_grey_blur)

        # combine change and foreground masks
        motion_mask = cv2.bitwise_or(diff_mask, fore_mask)

        # remove small pixel clusters
        contours, _ = cv2.findContours(
            motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in contours:
            if cv2.contourArea(c) < int(0.00005 * motion_mask.size):
                cv2.drawContours(motion_mask, [c], -1, (0.0,), -1)

        # resize previous track mask
        track_mask = cv2.resize(
            self.prev_track_mask,
            (_GREY_W, _GREY_H),
            interpolation=cv2.INTER_NEAREST,
        )

        # combine motion and previous detection masks into the next detection region
        search_mask = cv2.bitwise_or(motion_mask, track_mask)

        # store search mask and presence flag
        self._search_mask = cv2.resize(
            search_mask,
            (settings.FRAME_WIDTH, settings.FRAME_HEIGHT),
            interpolation=cv2.INTER_NEAREST,
        )
        self._has_search_area = (
            cv2.countNonZero(search_mask) / search_mask.size
        ) > 0.001

        # log detection duration
        utils.log_timing(logger, "Motion detection", start, self.hash)

        # log motion
        if self._has_search_area:
            logger.debug(f"({self.hash}) Has search area: {self._has_search_area}")

    @property
    def search_mask(self) -> np.ndarray:
        if not hasattr(self, "_search_mask"):
            self._identify_search_area()
        return self._search_mask

    @property
    def has_search_area(self) -> bool:
        if not hasattr(self, "_has_search_area"):
            self._identify_search_area()
        return self._has_search_area

    def _identify_search_bbox(self):

        logger.debug(f"({self.hash}) Identifying search bounding box")

        # identify bbox of the next detection region
        h, w = self.image.shape[:2]
        ys, xs = np.where(self.search_mask > 0)
        x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
        self._search_bbox = utils.expand_bbox_from_bounds(
            x_min, x_max, y_min, y_max, w, h, 0.1
        )

    @property
    def search_bbox(self) -> list:
        if not hasattr(self, "_search_bbox"):
            self._identify_search_bbox()
        return self._search_bbox

    def detect_objects(self) -> tuple[List[TrackFrame], bool]:
        track_frames = []

        # run detection if motion is present, active tracks exist, or on forced cadence
        if self.has_search_area or self.forced_detection_run:

            # log forced run
            if not self.has_search_area and self.forced_detection_run:
                logger.debug(f"({self.hash}) Forced object detection run")

            # start timing
            start = datetime.now()
            logger.debug(f"({self.hash}) Running object detection")
            did_run_detection = True

            # crop image to the search region
            if not self.has_search_area:
                image = self.image.copy()
                offsets = np.array([0, 0, 0, 0], dtype=np.int32)
            else:
                image = self.image[
                    self.search_bbox[1] : self.search_bbox[3],
                    self.search_bbox[0] : self.search_bbox[2],
                ].copy()
                offsets = np.array(
                    [
                        self.search_bbox[0],
                        self.search_bbox[1],
                        self.search_bbox[0],
                        self.search_bbox[1],
                    ],
                    dtype=np.int32,
                )

            # run model inference
            results = _model(
                image,
                imgsz=tuple(settings.DETECTION_IMGSZ),
                conf=settings.CONF,
                iou=settings.NMS_IOU_THRESHOLD,
                max_det=settings.MAX_DETS,
            )[0]

            # process detections
            for r in results.boxes:
                bbox = tuple(r.xyxy[0].cpu().numpy().astype(np.int32) + offsets)
                object_name = _model.names[int(r.cls[0].item())]
                frame_wh = (self.image.shape[1], self.image.shape[0])
                track_frames.append(
                    TrackFrame(
                        frame_hash=self.hash,
                        image=self.image,
                        bbox=utils.Bbox(xyxy=bbox, frame_wh=frame_wh),
                        object_name=object_name,
                        confidence=float(r.conf[0].item()),
                    )
                )

            # log detection duration
            utils.log_timing(logger, "Object detection", start, self.hash)

            # log detections
            if track_frames:
                logger.debug(
                    f"({self.hash}) Object(s) detected: {', '.join(f'{f.object_name} ({f.confidence:.2f})' for f in track_frames)}"
                )
        else:
            did_run_detection = False
        return track_frames, did_run_detection

    @property
    def processing_track_summaries(self) -> List[TrackSummary]:
        if not hasattr(self, "_processing_track_summaries"):
            raise RuntimeError(
                f"Processing track summaries not set yet for frame {self.hash}"
            )
        return self._processing_track_summaries

    @processing_track_summaries.setter
    def processing_track_summaries(
        self, processing_track_summaries: List[TrackSummary]
    ) -> None:
        self._processing_track_summaries = processing_track_summaries

    @property
    def recording_track_summaries(self) -> List[TrackSummary]:
        if not hasattr(self, "_recording_track_summaries"):
            raise RuntimeError(
                f"Recording track summaries not set yet for frame {self.hash}"
            )
        return self._recording_track_summaries

    @recording_track_summaries.setter
    def recording_track_summaries(
        self, recording_track_summaries: List[TrackSummary]
    ) -> None:
        self._recording_track_summaries = recording_track_summaries
