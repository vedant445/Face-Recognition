import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import tempfile
import os
import time

st.set_page_config(page_title="Face Mask Detector", page_icon="😷")
st.title("Face Mask Detection App")
st.write("Detect masks in images, videos, or live webcam snapshots.")

@st.cache_resource(show_spinner=True)
def load_mask_model():
    return tf.keras.models.load_model("mask_detector.model.h5")

mask_model = load_mask_model()

@st.cache_resource(show_spinner=True)
def load_face_detector():
    modelFile = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(modelFile)

face_detector = load_face_detector()

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

def detect_mask_frame(frame):
    faces = detect_faces(frame)
    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        if face_img.size == 0:
            continue
        face_img = cv2.resize(face_img, (160, 160))
        face_array = np.expand_dims(face_img / 255.0, axis=0)
        pred = mask_model.predict(face_array, verbose=0)[0]
        label = "Mask" if pred[0] > pred[1] else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame

# ----------------------------
# Input selection
# ----------------------------
option = st.radio("Choose input type:", ["Use Camera", "Upload Image", "Upload Video"])

# Webcam Snapshot
if option == "Use Camera":
    image_data = st.camera_input("Capture image using webcam:")
    if image_data:
        image = Image.open(image_data)
        frame = np.array(image.convert("RGB"))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result = detect_mask_frame(frame)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        st.image(result, channels="RGB", use_container_width=True)

# Image Upload
elif option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        frame = np.array(image.convert("RGB"))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result = detect_mask_frame(frame)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        st.image(result, channels="RGB", use_container_width=True)

# Video Upload
elif option == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video file...", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        st.info("Processing video... please wait ⏳")
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        progress = st.progress(0)
        frame_no = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1
            frame = detect_mask_frame(frame)
            out.write(frame)
            if frame_no % 5 == 0:
                progress.progress(min(frame_no / total_frames, 1.0))
        
        cap.release()
        out.release()
        progress.empty()

        time.sleep(1)  # wait for file flush
        st.success("✅ Video processing complete!")
        st.video(out_path)

        os.remove(tfile.name)
