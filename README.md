# 🛒 VisionCart

**Edge-Based Real-Time Automated Checkout System Using Embedded Computer Vision**

VisionCart is a computer-vision-powered retail checkout system that detects, identifies, and tallies items as a customer places them into a cart or checkout zone — no barcode scanning required. It's designed to run on low-power edge hardware (Raspberry Pi / NVIDIA Jetson class devices) instead of the camera-and-weight-sensor arrays that make "Just Walk Out"-style systems too expensive for small and medium retailers.

**Live demo:** _add your deployed URL here after following [Deployment](#-deployment-get-a-public-url) below_
`https://<your-app-name>.streamlit.app`

---

## Table of contents

- [How it works](#-how-it-works)
- [Repository structure](#-repository-structure)
- [Quickstart (local)](#-quickstart-local)
- [Deployment (get a public URL)](#-deployment-get-a-public-url)
- [Pushing this to GitHub](#-pushing-this-to-github)
- [Training your own model](#-training-your-own-model)
- [Architecture](#-architecture)
- [Tech stack](#-tech-stack)
- [Testing](#-testing)
- [Roadmap / limitations](#-roadmap--limitations)

---

## 🔍 How it works

1. A camera (webcam snapshot, uploaded photo, or uploaded video clip in this demo; an overhead-mounted camera in a real deployment) captures the checkout zone.
2. A YOLOv8 object-detection model runs on the frame(s) — on-device, not in the cloud.
3. Detections are cross-referenced against a **product catalog** and filtered by a **confidence threshold** to avoid false positives.
4. In video mode, a lightweight **centroid tracker** follows each item across frames so it's tallied into the cart exactly **once**, not once per frame.
5. The cart (running total + line items) updates instantly in the UI.

This mirrors the system architecture from the original project proposal:

```
Customer places item in Checkout Zone
              |
              v
   Overhead Camera Captures Video
              |
              v
     ┌───────────────────────┐
     │   Edge Computing Unit  │
     │ 1. Video Stream (OpenCV)│
     │ 2. Detection (YOLOv8)   │
     │ 3. ID & Confidence check│
     └───────────────────────┘
              |
              v
     Logic Layer Updates Cart DB
              |
              v
     Instant UI Update to User
```

## 📦 Repository structure

```
visioncart/
├── app/
│   ├── app.py               # Streamlit UI + orchestration
│   ├── detector.py          # YOLOv8 inference wrapper
│   ├── tracker.py           # Centroid tracker (temporal dedup)
│   ├── cart.py              # Cart/session logic (ERD's Cart_Items bridge table, in-memory)
│   └── product_catalog.py   # Product table (Product entity from the ERD)
├── docs/
│   └── erd.sql              # SQL schema matching the proposal's ERD
├── training/
│   └── README.md            # How to fine-tune YOLO on your own retail SKUs
├── tests/
│   └── test_cart_and_tracker.py
├── requirements.txt
├── Dockerfile
├── .streamlit/config.toml   # Theming
└── README.md
```

## 🚀 Quickstart (local)

Requires Python 3.10+.

```bash
git clone https://github.com/<your-username>/visioncart.git
cd visioncart
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). First run downloads the ~6 MB YOLOv8n weights automatically.

Three ways to feed it items:
- **Live Snapshot** — take a photo with your webcam via the browser
- **Upload Image** — drop in a product photo
- **Upload Video** — drop in a short clip; each item is tallied once via the tracker

> The demo recognizes the ~29 COCO classes listed in `app/product_catalog.py` (bottle, cup, banana, apple, backpack, laptop, etc.) as stand-ins for retail SKUs, since it ships without a custom-trained model. Point a phone camera at a water bottle, an apple, a backpack, a book, or a mug to see it work immediately. See [Training your own model](#-training-your-own-model) to recognize your actual products.

## ☁️ Deployment (get a public URL)

The fastest free option is **Streamlit Community Cloud** (built for exactly this).

1. Push this repo to GitHub (see next section).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **New app** → pick your `visioncart` repo → set:
   - **Main file path:** `app/app.py`
   - **Branch:** `main`
4. Click **Deploy**. In a couple of minutes you'll get a public URL like `https://visioncart-<random>.streamlit.app` — share that link (e.g. in your GitHub README, or with your instructor/reviewer).

**Alternative: Hugging Face Spaces**
1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space), SDK = **Streamlit**.
2. Either connect it to this GitHub repo or `git push` this folder to the Space's own git remote.
3. Make sure `app/app.py` is set as the app file in the Space settings (or move `app.py` to the repo root — Spaces expects it there by default).

**Alternative: Docker anywhere (Render / Railway / Fly.io / your own VPS)**
```bash
docker build -t visioncart .
docker run -p 8501:8501 visioncart
```
Push the image to any host that runs containers and exposes port 8501, and you have a public URL there too.

## 📤 Pushing this to GitHub

From inside the `visioncart/` folder:

```bash
git init
git add .
git commit -m "Initial commit: VisionCart edge checkout system"
git branch -M main
git remote add origin https://github.com/<your-username>/visioncart.git
git push -u origin main
```

(Create the empty repo on GitHub first at github.com/new — don't initialize it with a README there, to avoid a merge conflict with this one.)

## 🧠 Training your own model

The shipped app uses **stock YOLOv8n weights (COCO classes)** so it works immediately with no GPU and no dataset. To make it recognize your actual store's products, see **[`training/README.md`](training/README.md)** for the full pipeline: photographing your items → labeling in Roboflow → fine-tuning YOLOv8 → quantizing to ONNX/TensorRT for edge inference.

## 🏗️ Architecture

| Layer | Choice | Why |
|---|---|---|
| Detection model | YOLOv8n (Ultralytics) | Single-stage, high-FPS, easy to export to ONNX/TensorRT for edge devices |
| Video/image I/O | OpenCV | Standard for real-time frame handling and box visualization |
| Dedup across frames | Custom IoU + centroid tracker | Lightweight enough for CPU-only edge hardware; avoids double-counting one item across frames |
| Data layer | `Product` / `Cart_Session` / `Cart_Items` (see `docs/erd.sql`) | Matches the ERD from the original proposal; in-memory for the demo, drop-in SQLite/Postgres for production |
| UI | Streamlit | Fast to build, free public hosting via Streamlit Community Cloud |

## 🧰 Tech stack

Python · Streamlit · Ultralytics YOLOv8 · PyTorch · OpenCV · Pandas · Docker

## ✅ Testing

```bash
pip install pytest
pytest tests/ -v
```

Covers cart arithmetic, confidence-threshold filtering, unknown-class rejection, and — most importantly — that the tracker confirms an item exactly once across repeated frames and correctly distinguishes two simultaneous items.

## 🗺️ Roadmap / limitations

- **Demo uses COCO classes, not real SKUs.** Fine-tune per `training/README.md` for production use.
- **Tracker is IoU/centroid-based**, not a full ByteTrack/DeepSORT — swap it in if you need robustness to fast occlusion.
- **Cart is in-memory per Streamlit session**, not persisted to a real database — wire up `docs/erd.sql` with SQLite/Postgres for a production deployment with receipts/history.
- **No payment integration** — the proposal's "contactless payment terminal" step is out of scope for this reference implementation.

---

*Based on the original project proposal: "VisionCart: Edge-Based Real-Time Automated Checkout System Using Embedded Computer Vision."*
