FROM python:3.11-slim

# OpenCV needs these system libs even in "headless" mode
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Pre-download the YOLOv8n weights at build time so first request isn't slow
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

WORKDIR /app/app
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
