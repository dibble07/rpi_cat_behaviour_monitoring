from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import joblib
import ncnn
import numpy as np
import yaml  # type: ignore[import-untyped]

import utils
from config import settings
from tracking import TrackFrame, TrackSummary

logger = logging.getLogger(__name__)

_VERIFICATION_MODEL = joblib.load(Path("models") / "detection_verification.joblib")


class _Tensor:

    def __init__(self, values: np.ndarray) -> None:
        self._arr = np.asarray(values)

    def __getitem__(self, idx: int | slice) -> _Tensor:
        return _Tensor(self._arr[idx])

    def cpu(self) -> _Tensor:
        """Mimic torch's Tensor.cpu for ultralytics-style results"""
        return self

    def numpy(self) -> np.ndarray:
        """Mimic torch's Tensor.numpy for ultralytics-style results"""
        return self._arr

    def item(self) -> float:
        """Mimic torch's Tensor.item for ultralytics-style results"""
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
        iom: float = 0.9,
        max_det: int = 300,
    ) -> None:
        # load static model config and class names
        model_dir = Path(model)
        self.conf, self.iou, self.iom, self.max_det = conf, iou, iom, max_det
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
        conf: float | None = None,
        iou: float | None = None,
        iom: float | None = None,
        max_det: int | None = None,
        verbose: bool = True,
    ) -> list[SimpleNamespace]:
        """Detect objects in an image, mimicking the ultralytics predict interface"""

        # resolve thresholds and model input size
        eff_conf = self.conf if conf is None else conf
        eff_iou = self.iou if iou is None else iou
        eff_iom = self.iom if iom is None else iom
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

        # arrange output as one row per candidate box
        pred = np.asarray(out0)
        if pred.shape[0] == 6:
            pred = pred.T

        # split class logits/probabilities from box channels
        cls_scores = pred[:, 4:]
        cls_idx = cls_scores.argmax(axis=1)
        cls_conf = cls_scores.max(axis=1)

        # drop low-confidence
        valid = cls_conf > eff_conf
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
        keep = []
        for cls in np.unique(cls_idx):
            idx = np.where(cls_idx == cls)[0]
            picked = utils.nms_boxes(boxes[idx], cls_conf[idx], eff_iou, eff_iom)
            if len(picked):
                keep.extend(idx[picked].tolist())
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
_MODEL = YoloNcnn(Path("models") / settings.MODEL_DETECTION_PATH)
_ = _MODEL(
    np.zeros((settings.FRAME_HEIGHT, settings.FRAME_WIDTH, 3), dtype=np.uint8),
    imgsz=tuple(settings.DETECTION_IMGSZ),
    conf=settings.CONF,
    iou=settings.NMS_IOU_THRESHOLD,
    iom=settings.NMS_IOM_THRESHOLD,
    max_det=settings.MAX_DETS,
)

# define background subtractor
_back_sub = cv2.createBackgroundSubtractorMOG2(
    history=settings.BACKGROUND_HISTORY, detectShadows=False
)


class Frame:
    """Store frame image, timestamp, and supplementary processing state"""

    def __init__(
        self,
        timestamp: datetime,
        image: np.ndarray,
        prev_frame: Frame | None,
        prev_track_mask: np.ndarray,
        forced_detection_run: bool,
    ) -> None:

        # store frame inputs
        self.timestamp = timestamp
        self.image = np.ascontiguousarray(image)
        self.prev_track_mask = prev_track_mask
        self.forced_detection_run = forced_detection_run

        # blur and hash the image
        start = datetime.now()
        self.image_grey_blur = utils.blur_image(self.image)
        self.hash = utils.hash_image(self.image_grey_blur)
        utils.log_timing(logger, "Blur and hash", start, self.hash)

        # keep the previous blurred image for motion detection
        if prev_frame is None:
            logger.warning(f"({self.hash}) No previous frame provided")
            self.prev_image_grey_blur = np.zeros_like(self.image_grey_blur)
        else:
            self.prev_image_grey_blur = prev_frame.image_grey_blur.copy().astype(
                np.uint8
            )

    def _identify_search_area(self) -> None:
        """Identify search area via frame differencing, background subtraction and previous tracks"""
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
            (self.image_grey_blur.shape[1], self.image_grey_blur.shape[0]),
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
        """Mask of regions with motion or previous tracks"""
        if not hasattr(self, "_search_mask"):
            self._identify_search_area()
        return self._search_mask

    @property
    def has_search_area(self) -> bool:
        """Whether enough of the frame needs searching to run detection"""
        if not hasattr(self, "_has_search_area"):
            self._identify_search_area()
        return self._has_search_area

    @property
    def search_bbox(self) -> list[int]:
        """Padded bounding box around the search mask"""

        # identify bbox of the next detection region
        if not hasattr(self, "_search_bbox"):
            logger.debug(f"({self.hash}) Identifying search bounding box")
            h, w = self.image.shape[:2]
            ys, xs = np.where(self.search_mask > 0)
            x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
            self._search_bbox = utils.expand_bbox_from_bounds(
                x_min, x_max, y_min, y_max, w, h, 0.1
            )
        return self._search_bbox

    def detect_objects(self) -> tuple[list[TrackFrame], bool]:
        """Detect objects within the search region and wrap them for tracking"""
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
            results = _MODEL(
                image,
                imgsz=tuple(settings.DETECTION_IMGSZ),
                conf=settings.CONF,
                iou=settings.NMS_IOU_THRESHOLD,
                iom=settings.NMS_IOM_THRESHOLD,
                max_det=settings.MAX_DETS,
            )[0]

            # process detections
            for r in results.boxes:
                bbox = tuple(r.xyxy[0].cpu().numpy().astype(np.int32) + offsets)
                object_name = _MODEL.names[int(r.cls[0].item())]
                frame_wh = (self.image.shape[1], self.image.shape[0])
                track_frames.append(
                    TrackFrame(
                        frame_hash=self.hash,
                        image=self.image,
                        bbox=utils.Bbox(xyxy=bbox, frame_wh=frame_wh),
                        object_name=object_name,
                        confidence=r.conf[0].item(),
                    )
                )

            # log detection duration
            utils.log_timing(logger, "Object detection", start, self.hash)

            # verify detections after object detection timing has completed
            if settings.VERIFY_DETECTIONS and track_frames:
                embeddings = [f.roi_embedding for f in track_frames]

                # compare verifier predictions with detected classes
                start = datetime.now()
                verified_track_frames = []
                for frame, emb in zip(track_frames, embeddings):
                    if (
                        _VERIFICATION_MODEL.predict(emb[None, :])[0]
                        == frame.object_name
                    ):
                        verified_track_frames.append(frame)
                track_frames = verified_track_frames
                utils.log_timing(logger, "Detection verification", start, self.hash)

            # log detections
            if track_frames:
                logger.debug(
                    f"({self.hash}) Object(s) detected: {', '.join(f'{f.object_name} ({f.confidence:.2f})' for f in track_frames)}"
                )
        else:
            did_run_detection = False
        return track_frames, did_run_detection

    @property
    def proc_ids(self) -> list[int]:
        """Ids of tracks active when this frame was processed"""
        if not hasattr(self, "_proc_ids"):
            raise RuntimeError(
                f"Processing track ids not set yet for frame {self.hash}"
            )
        return self._proc_ids

    @proc_ids.setter
    def proc_ids(self, proc_ids: list[int]) -> None:
        self._proc_ids = proc_ids

    @property
    def recording_track_summaries(self) -> list[TrackSummary]:
        """Track summaries captured when this frame was recorded"""
        if not hasattr(self, "_recording_track_summaries"):
            raise RuntimeError(
                f"Recording track summaries not set yet for frame {self.hash}"
            )
        return self._recording_track_summaries

    @recording_track_summaries.setter
    def recording_track_summaries(
        self, recording_track_summaries: list[TrackSummary]
    ) -> None:
        self._recording_track_summaries = recording_track_summaries
