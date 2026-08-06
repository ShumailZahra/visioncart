# Training a custom retail-item model

The shipped app uses stock YOLOv8n weights (COCO classes) so it runs
immediately with zero setup. This is how you replace that with a model
trained on your **own** 15-20 retail SKUs, following the methodology in the
project proposal.

## 1. Collect & label data (Roboflow)

1. Photograph each of your 15-20 items under varying lighting, angles, and
   partial occlusion (the proposal calls for this explicitly -- aim for
   80-150 images per item).
2. Create a project at [roboflow.com](https://roboflow.com), upload the
   images, and draw bounding boxes per item.
3. Use Roboflow's augmentation step (brightness/rotation/blur jitter) to
   synthetically expand the dataset.
4. Export in **YOLOv8** format -- Roboflow generates a ready-to-use
   `data.yaml` plus `train/`, `valid/`, `test/` image+label folders.

Place the export at `training/dataset/` (this path is git-ignored).

## 2. Fine-tune

```bash
pip install ultralytics
yolo detect train \
  data=training/dataset/data.yaml \
  model=yolov8n.pt \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  name=visioncart_custom
```

Weights land at `runs/detect/visioncart_custom/weights/best.pt`.

## 3. Quantize for edge inference (TensorRT / ONNX)

```bash
# ONNX Runtime (portable, CPU or GPU)
yolo export model=runs/detect/visioncart_custom/weights/best.pt format=onnx

# TensorRT (NVIDIA Jetson / GPU boxes) -- run ON the target device
yolo export model=runs/detect/visioncart_custom/weights/best.pt format=engine half=True
```

`half=True` gives FP16 quantization; add `int8=True data=training/dataset/data.yaml`
for INT8 if you need the extra speed and can tolerate the accuracy hit.

## 4. Plug it into the app

```bash
export VISIONCART_MODEL_PATH=/path/to/best.pt
streamlit run app/app.py
```

Then update `app/product_catalog.py` so every `visual_class_label` matches
a class name from your new `data.yaml`, with real prices/SKUs from your
POS system.

## Target hardware notes (per proposal)

| Device | Expected FPS (yolov8n, 640px, FP16) |
|---|---|
| Raspberry Pi 4 + Coral/accelerator | ~15-25 FPS |
| NVIDIA Jetson Nano | ~20-30 FPS |
| NVIDIA Jetson Orin Nano | 60+ FPS |

Benchmark on your actual device -- these are ballpark figures, not guarantees.
