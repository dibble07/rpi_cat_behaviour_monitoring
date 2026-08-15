from __future__ import annotations

import importlib
import importlib.util
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np


def _load_model_class():
    try:
        module = importlib.import_module("inference_ext")
        return getattr(module, "Model")
    except ModuleNotFoundError as first_exc:
        repo_root = Path(__file__).resolve().parent.parent
        wheels_dir = repo_root / "inference_ext" / "target" / "wheels"
        candidates = sorted(
            wheels_dir.glob("inference_ext-*.whl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for wheel_path in candidates:
            extract_dir = wheels_dir / ".runtime" / wheel_path.stem
            try:
                with zipfile.ZipFile(wheel_path) as zf:
                    so_members = [
                        n
                        for n in zf.namelist()
                        if n.endswith((".so", ".pyd", ".dylib"))
                        and "/inference_ext" in f"/{n}"
                    ]
                    if not so_members:
                        continue
                    so_member = so_members[0]
                    extract_dir.mkdir(parents=True, exist_ok=True)
                    extracted_path = extract_dir / Path(so_member).name
                    with zf.open(so_member) as src, open(extracted_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)

                spec = importlib.util.spec_from_file_location(
                    "inference_ext", extracted_path
                )
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules["inference_ext"] = module
                spec.loader.exec_module(module)
                return getattr(module, "Model")
            except Exception:
                continue

        raise ModuleNotFoundError(
            "Rust extension 'inference_ext' not found. Expected a prebuilt wheel at "
            f"{wheels_dir}. Commit/copy the correct cp/manylinux aarch64 wheel and retry."
        ) from first_exc


_Model = _load_model_class()


class _TensorLike(np.ndarray):
    """numpy subclass with .cpu() / .numpy() / .item() for torch tensor compatibility."""

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def item(self):
        return self.flat[0]


def _to_tensor(a) -> _TensorLike:
    return np.asarray(a, dtype=np.float32).view(_TensorLike)


class _Box:
    """Single detection, mirroring one row of ultralytics Boxes."""

    __slots__ = ("xyxy", "conf", "cls")

    def __init__(
        self, x1: float, y1: float, x2: float, y2: float, conf: float, cls: float
    ) -> None:
        self.xyxy = _to_tensor([[x1, y1, x2, y2]])  # (1, 4)
        self.conf = _to_tensor([conf])  # (1,)
        self.cls = _to_tensor([cls])  # (1,)


class _Result:
    """Detection result for one image, mirroring ultralytics Results."""

    def __init__(self, names: Dict[int, str], raw: List) -> None:
        self.names = names
        self.boxes: List[_Box] = [_Box(*r) for r in raw]


class YOLOOnnx:
    """Drop-in replacement for ultralytics.YOLO for detection tasks via ONNX Runtime."""

    def __init__(
        self,
        model: Union[str, Path],
        task: str = "detect",
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
    ) -> None:
        onnx_path = Path(model)
        if not onnx_path.is_file():
            raise FileNotFoundError(f"Model file not found: {onnx_path}")
        self._onnx_path = onnx_path
        self.task = "detect"
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self._rust = _Model(str(self._onnx_path), conf, iou, max_det)
        self.names: Dict[int, str] = self._rust.names()

    def predict(
        self,
        source: Union[str, Path, np.ndarray],
        imgsz: int = 640,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: Optional[int] = None,
        verbose: bool = True,
        **_,
    ) -> List[_Result]:
        """Run detection on an image file path or BGR numpy array."""
        eff_conf = conf if conf is not None else self.conf
        eff_iou = iou if iou is not None else self.iou
        eff_max_det = max_det if max_det is not None else self.max_det
        if (eff_conf, eff_iou, eff_max_det) != (self.conf, self.iou, self.max_det):
            self._rust = _Model(str(self._onnx_path), eff_conf, eff_iou, eff_max_det)
            self.conf, self.iou, self.max_det = eff_conf, eff_iou, eff_max_det

        if isinstance(source, np.ndarray):
            rgb = np.ascontiguousarray(source[:, :, ::-1])  # BGR → RGB
            h, w = source.shape[:2]
            raw = self._rust.predict_rgb(h, w, rgb.tobytes())
        else:
            raw = self._rust.predict_path(str(source))
        return [_Result(self.names, raw)]

    def __call__(self, source, **kw) -> List[_Result]:
        return self.predict(source, **kw)
