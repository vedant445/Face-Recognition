# 2️⃣ Image Upload (MTCNN adjusted for masked faces)
elif option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        frame = np.array(image.convert("RGB"))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        try:
            detections = face_detector.detect_faces(rgb_frame)
        except Exception:
            detections = []

        if not detections:
            st.warning("No face detected. Try uploading a clearer image.")
        else:
            for det in detections:
                x, y, w, h = det["box"]
                x, y = max(0, x), max(0, y)
                w, h = abs(w), abs(h)

                # Increase padding for masked faces
                pad = 20
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)

                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                face_img = cv2.resize(face_img, (160, 160))
                face_array = np.expand_dims(face_img, axis=0) / 255.0
                pred = mask_model.predict(face_array, verbose=0)[0]

                label = "Mask" if pred[0] > pred[1] else "No Mask"
                confidence = max(pred)
                color = (0, 255, 0) if label == "Mask" else (255, 0, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ({confidence*100:.1f}%)",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            st.image(frame, channels="RGB", use_container_width=True)
            st.success("✅ Detection complete.")
