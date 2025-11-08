import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import tempfile
import os

st.set_page_config(page_title="Face Mask Detector", page_icon="😷")
st.title("Face Mask Detection App")
st.write("Detect masks in images, videos, or live webcam snapshots.")

@st.cache_resource(show_spinner=True)
def load_mask_model():
    return tf.keras.models.load_model("mask_detector.model.h5")

mask_model = load_mask_model()

# Load OpenCV face detector (much faster)
@st.cache_resource(show_spinner=True)
def load_opencv_face_detector():
    modelFile = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(modelFile)

face_detector = load_opencv_face_detector()

def detect_faces_opencv(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    return faces

def detect_mask_frame(frame, confidence_threshold=0.6):
    results = []
    faces = detect_faces_opencv(frame)
    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        if face_img.size == 0:
            continue
        face_img = cv2.resize(face_img, (160, 160))
        face_array = np.expand_dims(face_img, axis=0) / 255.0
        pred = mask_model.predict(face_array, verbose=0)[0]
        if max(pred) < confidence_threshold:
            continue
        label = "Mask" if pred[0] > pred[1] else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (255, 0, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        results.append(label)
    return frame, results

# ----------------------------
# Input selection
# ----------------------------
option = st.radio("Choose input type:", ["Use Camera", "Upload Image", "Upload Video"])

# Webcam Snapshot
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

# Image Upload
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

# Video Upload
elif option == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video file...", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        st.info("Processing video... please wait ⏳")
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 24)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 3 != 0:
                continue  # Skip some frames for speed
            processed_frame, _ = detect_mask_frame(frame)
            out.write(processed_frame)

        cap.release()
        out.release()
        st.success("✅ Video processing complete!")
        st.video(output_path)
        os.remove(tfile.name)
