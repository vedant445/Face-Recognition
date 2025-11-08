import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from mtcnn import MTCNN
import tempfile
import os
import time

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="Face Mask Detector", page_icon="😷")
st.title("Face Mask Detection App")
st.write("Detect masks in images, videos, or live webcam snapshots.")

# ----------------------------
# Model Loaders
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_mask_model():
    return tf.keras.models.load_model("mask_detector.model.h5")

@st.cache_resource(show_spinner=True)
def load_mtcnn():
    return MTCNN()

mask_model = load_mask_model()
face_detector = load_mtcnn()

# ----------------------------
# Helper Functions
# ----------------------------
def detect_faces_mtcnn(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        results = face_detector.detect_faces(rgb_frame)
    except Exception:
        return []
    faces = []
    for res in results:
        x, y, w, h = res['box']
        x, y = max(0, x), max(0, y)
        faces.append((x, y, w, h))
    return faces


def detect_mask_frame(frame, confidence_threshold=0.6, min_face_size=50, pad=10):
    results = []
    faces = detect_faces_mtcnn(frame)
    for (x, y, w, h) in faces:
        if w < min_face_size or h < min_face_size:
            continue
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            continue
        face_img = cv2.resize(face_img, (160, 160))
        face_array = np.expand_dims(face_img, axis=0) / 255.0
        pred = mask_model.predict(face_array, verbose=0)[0]
        if max(pred) < confidence_threshold:
            continue
        label = "Mask" if pred[0] > pred[1] else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        results.append(label)
    return frame, results


# ----------------------------
# Input Options
# ----------------------------
option = st.radio("Choose input type:", ["Use Camera", "Upload Image", "Upload Video"])

# 1️⃣ Webcam Snapshot
if option == "Use Camera":
    image_data = st.camera_input("Capture image using webcam:")
    if image_data:
        image = Image.open(image_data)
        frame, results = detect_mask_frame(np.array(image.convert("RGB")))
        st.image(frame, channels="RGB", use_container_width=True)
        if results:
            st.success(f"Detected: {results}")
        else:
            st.warning("No face detected.")

# 2️⃣ Image Upload
elif option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        frame, results = detect_mask_frame(np.array(image.convert("RGB")))
        st.image(frame, channels="RGB", use_container_width=True)
        if results:
            st.success(f"Detected: {results}")
        else:
            st.warning("No face detected.")

# 3️⃣ Video Upload with Live Preview
elif option == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video file...", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        st.info("Processing video... please wait ⏳")

        # Save uploaded file to temporary path
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("❌ Could not open video file. Try re-uploading.")
            st.stop()

        fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output video writer (.avi more stable)
        out_path_avi = tempfile.NamedTemporaryFile(delete=False, suffix=".avi").name
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(out_path_avi, fourcc, fps, (w, h))

        # Streamlit live preview setup
        progress_bar = st.progress(0)
        status_text = st.empty()
        video_placeholder = st.empty()

        frame_no = 0
        frames_processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1

            processed_frame, _ = detect_mask_frame(frame)
            out.write(processed_frame)
            frames_processed += 1

            # Show live preview
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, caption=f"Processing frame {frame_no}/{total_frames}", use_container_width=True)

            # Progress bar update
            progress_bar.progress(min(frame_no / total_frames, 1.0))
            status_text.text(f"Processed {frame_no}/{total_frames} frames...")

        cap.release()
        out.release()

        progress_bar.empty()
        status_text.text("✅ Video processing complete!")

        if os.path.exists(out_path_avi) and os.path.getsize(out_path_avi) > 10000:
            st.video(out_path_avi)
            st.success(f"✅ Processed {frames_processed} frames successfully!")
        else:
            st.error("⚠️ Processed video appears empty. Try a different file.")

        # Clean up
        os.remove(tfile.name)

