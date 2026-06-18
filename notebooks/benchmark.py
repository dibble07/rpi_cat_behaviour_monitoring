import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from ultralytics import YOLO

from config import settings

# Load a YOLO26n PyTorch model
model = YOLO("yolo26n.pt")

# Benchmark YOLO26n speed and accuracy on the COCO128 dataset for all export formats
_ = model.benchmark(data="coco128.yaml", imgsz=settings.IMGSZ, half=True, device="cpu")
