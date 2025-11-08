import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from mtcnn import MTCNN
import tempfile
import os

st.set_page_config(page_title="Face Mask Detector", page_icon="😷")
st.title("Face Mask Detection App")
st.write("Detect masks in images, videos, or live webcam snapshots.")

# ----------------------------
# Load Models
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_mask_model():
    model = tf.keras.models.load_model("mask_detector.model.h5")
    return model

mask_model = load_mask_model()

@st.cache_resource(show_spinner=True)
def load_mtcnn():
    return MTCNN()

face_detector = load_mtcnn()

# ----------------------------
# Helper Functions
# ----------------------------
def detect_faces_mtcnn(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.detect_faces(rgb_frame)
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
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        results.append(label)
    return frame, results

# ----------------------------
# Input Options
# ----------------------------
option = st.radio("Choose input type:", ["Use Camera", "Upload Image", "Upload Video"])

# ----------------------------
# Option 1: Camera Snapshot
# ----------------------------
if option == "Use Camera":
    st.write("Capture image using your webcam below 👇")
    image_data = st.camera_input("Take a picture")

    if image_data:
        image = Image.open(image_data)
        frame, results = detect_mask_frame(np.array(image.convert("RGB")))
        st.image(frame, channels="RGB", use_container_width=True)
        if results:
            st.success(f"Detected: {results}")
        else:
            st.warning("No face detected.")

# ----------------------------
# Option 2: Image Upload
# ----------------------------
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

# ----------------------------
# Option 3: Video Upload
# ----------------------------
elif option == "Upload Video":
    video_file = st.file_uploader("Upload a video file...", type=["mp4", "avi", "mov", "mkv"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        frame_skip = 3  # process every 3rd frame for speed

        st.info("Processing video... please wait ⏳")
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame = cv2.resize(frame, (640, 480))
            frame, _ = detect_mask_frame(frame)
            stframe.image(frame, channels="BGR", use_container_width=True)

        cap.release()
        st.success("✅ Video processed successfully!")
        os.remove(tfile.name)
