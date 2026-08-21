"""
detector.py
-----------
Runs YOLOv8n inference via **ONNX Runtime** rather than PyTorch/ultralytics.

Why ONNX Runtime and not the ultralytics/torch path used during
development: `ultralytics` pulls in PyTorch as a dependency, and a plain
`pip install torch` on Linux resolves to the full CUDA build (~2GB+ of
torch + nvidia-cublas/cudnn/nccl/triton) even though this app only ever
does CPU inference. That's what blows past Streamlit Community Cloud's
build disk budget. ONNX Runtime's CPU package is ~30MB with zero CUDA
dependencies -- and it's exactly the deployment path the project proposal
itself specifies (Section 4.2: "TensorRT or ONNX Runtime ... for high-FPS
inference on edge hardware").

The shipped assets/yolov8n.onnx was exported once, offline, with NMS baked
into the graph (`model.export(format="onnx", nms=True)`), so this file
does not need to implement non-max suppression itself -- the model's raw
output is already a clean (1, 300, 6) tensor of
[x1, y1, x2, y2, confidence, class_id] rows.

To use your own fine-tuned model instead (see /training/README.md):
    yolo export model=best.pt format=onnx nms=True opset=12
then point MODEL_PATH at the resulting best.onnx and update
product_catalog.py's visual_class_label values to match your new classes.
"""

from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np

MODEL_PATH = os.environ.get(
    "VISIONCART_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "assets", "yolov8n.onnx"),
)
INPUT_SIZE = 640

# COCO's 80 class names, in the index order YOLOv8 was trained with.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


@lru_cache(maxsize=1)
def load_session():
    import onnxruntime as ort

    return ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


def _letterbox(frame: np.ndarray, size: int = INPUT_SIZE):
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top


def run_inference(frame: np.ndarray, confidence: float, allowed_classes: set[str]):
    """
    Runs one forward pass and returns:
      annotated_frame (np.ndarray, BGR),
      detections: list of (class_name, (x1,y1,x2,y2), confidence)
    Only classes in `allowed_classes` (the retail catalog) are kept -- this
    is the "Confidence Threshold Filtering" + catalog cross-reference step.
    """
    session = load_session()
    canvas, scale, pad_x, pad_y = _letterbox(frame)

    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, normalize
    blob = np.transpose(blob, (2, 0, 1))[None, ...]  # HWC -> NCHW

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: blob})[0]  # (1, 300, 6)
    rows = output[0]

    annotated = frame.copy()
    detections = []
    for x1, y1, x2, y2, conf, cls_id in rows:
        conf = float(conf)
        if conf < confidence:
            continue
        cls_id = int(cls_id)
        if cls_id < 0 or cls_id >= len(COCO_CLASSES):
            continue
        cls_name = COCO_CLASSES[cls_id]

        # undo letterbox padding/scaling to map back to original frame coords
        ox1 = (x1 - pad_x) / scale
        oy1 = (y1 - pad_y) / scale
        ox2 = (x2 - pad_x) / scale
        oy2 = (y2 - pad_y) / scale
        box = (max(0.0, ox1), max(0.0, oy1), ox2, oy2)

        color = (0, 200, 0) if cls_name in allowed_classes else (160, 160, 160)
        cv2.rectangle(annotated, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(annotated, label, (int(box[0]), max(0, int(box[1]) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if cls_name in allowed_classes:
            detections.append((cls_name, box, conf))

    return annotated, detections
