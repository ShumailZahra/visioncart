"""
detector.py
-----------
Wraps an Ultralytics YOLO model behind the interface the rest of the app
needs. This is "Step 2/3: Object Detection Inference + Item ID & Confidence
Verification" from the System Architecture diagram in the proposal.

Model choice
------------
Ships with YOLOv8n (nano) pretrained on COCO so the project runs
out-of-the-box with no GPU and no custom dataset. That satisfies the
*architecture* the proposal describes (single-stage detector, ONNX/TensorRT
exportable, edge-deployable). To make it a *true* retail SKU recognizer,
fine-tune on your own item photos:

    yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640

then point MODEL_PATH below at the resulting best.pt, and update
product_catalog.py's visual_class_label values to match your new class
names. See /training/README.md in this repo for the full fine-tuning +
export-to-ONNX/TensorRT walkthrough referenced in the proposal.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

MODEL_PATH = os.environ.get("VISIONCART_MODEL_PATH", "yolov8n.pt")


@lru_cache(maxsize=1)
def load_model():
    from ultralytics import YOLO

    return YOLO(MODEL_PATH)


def run_inference(frame: np.ndarray, confidence: float, allowed_classes: set[str]):
    """
    Runs one forward pass and returns:
      annotated_frame (np.ndarray, BGR),
      detections: list of (class_name, (x1,y1,x2,y2), confidence)
    Only classes in `allowed_classes` (the retail catalog) are kept -- this
    is the "Confidence Threshold Filtering" + catalog cross-reference step.
    """
    model = load_model()
    results = model.predict(frame, conf=confidence, verbose=False)[0]

    detections = []
    names = results.names
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        conf = float(box.conf[0])
        if cls_name not in allowed_classes:
            continue
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        detections.append((cls_name, (x1, y1, x2, y2), conf))

    annotated = results.plot()  # BGR np.ndarray with boxes drawn (all classes, for visual context)
    return annotated, detections
