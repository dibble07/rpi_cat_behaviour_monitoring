from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import ncnn
import numpy as np
import yaml  # type: ignore[import-untyped]


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


class YOLO_NCNN:

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
