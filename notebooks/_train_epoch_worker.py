import logging
import os
import re
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DATASETS_DIR = os.path.join(_PROJECT_ROOT, "datasets")
_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "yolo26n_subclass.pt")

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import psutil
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER as _ultralytics_logger

_LAYER_ROW_RE = re.compile(r"^\s+\d+\s+")


class _SuppressVerboseFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "freezing layer" in msg.lower():
            return False
        if "from  n    params" in msg:  # layer table header
            return False
        if _LAYER_ROW_RE.match(msg):  # layer table rows
            return False
        if "summary:" in msg:  # model summary line
            return False
        if msg.startswith("Transferred "):  # weight transfer info
            return False
        return True


def run(eval_str, imgsz, device_str):
    _ultralytics_logger.addFilter(_SuppressVerboseFilter())

    def get_comp_state():
        mps_alloc = torch.mps.current_allocated_memory() / 1e9
        mps_driver = torch.mps.driver_allocated_memory() / 1e9
        ram = psutil.virtual_memory()
        ram_used = (ram.total - ram.available) / 1e9
        ram_total = ram.total / 1e9
        swap = psutil.swap_memory()
        swap_used = swap.used / 1e9
        return f"MPS alloc= {mps_alloc:.2f} GB | MPS driver= {mps_driver:.2f} GB | RAM= {ram_used:.1f}/{ram_total:.1f} GB ({ram.percent:.1f}%) | swap= {swap_used:.2f} GB"

    def on_train_epoch_end(trainer):
        print(f"Epoch {trainer.epoch+1} END:   {get_comp_state()}", flush=True)

    def on_train_epoch_start(trainer):
        print(f"Epoch {trainer.epoch+1} START: {get_comp_state()}", flush=True)

    model = YOLO(_MODEL_PATH)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    model.train(
        data=os.path.join(_DATASETS_DIR, f"coco{eval_str}_subclass_downsample.yaml"),
        epochs=1,
        batch=8,
        imgsz=imgsz,
        cache=False,
        device=device_str,
        workers=0,
        amp=True,
        freeze=23,
        lr0=1e-4,
        lrf=1,
        warmup_epochs=0,
        plots=False,
        mosaic=1,
    )
    model.save(_MODEL_PATH)
