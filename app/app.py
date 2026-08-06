"""
VisionCart - Edge-Based Real-Time Automated Checkout System
=============================================================
Streamlit front-end. Run with:  streamlit run app.py

Three input modes map onto the proposal's "Overhead Camera Captures Video"
step, adapted to what a browser-based demo can actually capture:

  * Live Snapshot  - st.camera_input (single frame from your webcam)
  * Upload Image   - single product photo
  * Upload Video   - short clip, processed frame-by-frame with the
                     CentroidTracker so items are only added to the cart
                     once (the "Dynamic Cart Tallying" requirement)

All three funnel through the same detector.run_inference() +
cart.Cart.add_by_class_name() path, mirroring the System Architecture
diagram: Video -> Edge Computing Unit -> Logic Layer -> Cart DB -> UI.
"""

import time
import uuid

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from cart import Cart
from detector import run_inference
from product_catalog import get_tracked_class_names
from tracker import CentroidTracker

st.set_page_config(page_title="VisionCart", page_icon="🛒", layout="wide")

# ---------------------------------------------------------------- session --
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "cart" not in st.session_state:
    st.session_state.cart = Cart(session_id=st.session_state.session_id)
if "tracker" not in st.session_state:
    st.session_state.tracker = CentroidTracker()

cart: Cart = st.session_state.cart
tracker: CentroidTracker = st.session_state.tracker
TRACKED_CLASSES = set(get_tracked_class_names())

# ------------------------------------------------------------------- UI --
st.title("🛒 VisionCart")
st.caption("Edge-Based Real-Time Automated Checkout System Using Embedded Computer Vision")

with st.sidebar:
    st.header("Session")
    st.write(f"**Session ID:** `{st.session_state.session_id}`")
    confidence = st.slider("Confidence threshold", 0.1, 0.95, 0.5, 0.05)
    min_hits = st.slider("Frames to confirm an item (video mode)", 1, 10, 3, 1)
    tracker.min_hits_to_count = min_hits

    st.divider()
    st.header("🧾 Cart")
    rows = cart.as_rows()
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.metric("Total", f"PKR {cart.total_amount:,.0f}", f"{cart.item_count} item(s)")
    else:
        st.info("Cart is empty. Detect an item to add it.")

    if st.button("🗑️ Clear cart / New session", use_container_width=True):
        cart.clear()
        tracker.reset()
        st.rerun()

tab_snap, tab_img, tab_vid, tab_log = st.tabs(
    ["📷 Live Snapshot", "🖼️ Upload Image", "🎞️ Upload Video", "📜 Activity Log"]
)

# ---------------------------------------------------------- snapshot mode --
with tab_snap:
    st.write("Take a photo of an item as if placing it in the checkout zone.")
    snap = st.camera_input("Checkout zone camera", label_visibility="collapsed")
    if snap is not None:
        image = Image.open(snap).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        annotated, detections = run_inference(frame_bgr, confidence, TRACKED_CLASSES)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detection result", use_container_width=True)

        if not detections:
            st.warning("No catalog items detected above the confidence threshold.")
        for cls_name, box, conf in detections:
            product = cart.add_by_class_name(cls_name, conf, confidence)
            if product:
                st.success(f"Added **{product.display_name}** to cart (confidence {conf:.0%})")
        if detections:
            st.rerun()

# ------------------------------------------------------------- image mode --
with tab_img:
    st.write("Upload a photo of a single item (or a shelf of items).")
    uploaded = st.file_uploader("Item photo", type=["jpg", "jpeg", "png"], key="img_uploader")
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        annotated, detections = run_inference(frame_bgr, confidence, TRACKED_CLASSES)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detection result", use_container_width=True)

        if not detections:
            st.warning("No catalog items detected above the confidence threshold.")
        for cls_name, box, conf in detections:
            product = cart.add_by_class_name(cls_name, conf, confidence)
            if product:
                st.success(f"Added **{product.display_name}** to cart (confidence {conf:.0%})")

# ------------------------------------------------------------- video mode --
with tab_vid:
    st.write(
        "Upload a short clip of items passing the camera. Frames are processed "
        "sequentially through the centroid tracker so each physical item is "
        "only tallied into the cart once, even though it appears in many frames."
    )
    video_file = st.file_uploader("Checkout zone clip", type=["mp4", "mov", "avi"], key="vid_uploader")
    frame_skip = st.number_input("Process every Nth frame (speed vs. accuracy)", 1, 10, 2)

    if video_file is not None and st.button("▶️ Process video"):
        tmp_path = f"/tmp/{uuid.uuid4().hex}_{video_file.name}"
        with open(tmp_path, "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture(tmp_path)
        frame_placeholder = st.empty()
        progress = st.progress(0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_idx = 0
        tracker.reset()

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            annotated, detections = run_inference(frame, confidence, TRACKED_CLASSES)
            confirmed = tracker.update(detections)
            for track in confirmed:
                # use the highest-confidence detection this frame for that class
                match_conf = max((c for cls, _, c in detections if cls == track.class_name), default=confidence)
                product = cart.add_by_class_name(track.class_name, match_conf, confidence)
                if product:
                    st.toast(f"Added {product.display_name} to cart", icon="🛒")

            frame_placeholder.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption=f"Frame {frame_idx}/{total_frames}",
                use_container_width=True,
            )
            progress.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        st.success("Video processed. Check the cart in the sidebar.")
        st.rerun()

# --------------------------------------------------------------- log tab --
with tab_log:
    st.write("Raw event log for this session (mirrors the Logic Layer -> Cart DB updates).")
    if cart.log:
        st.code("\n".join(cart.log), language=None)
    else:
        st.info("No activity yet.")

st.divider()
st.caption(
    "Demo note: ships with YOLOv8n pretrained on COCO so it runs without a GPU or custom "
    "dataset. Swap in a retail-fine-tuned model per the proposal's methodology to recognize "
    "real SKUs -- see the README's 'Training your own model' section."
)
