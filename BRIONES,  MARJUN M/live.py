import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import time
import os

st.title("🎥 Live Object Detection, Counting & Alerts")

SAVE_DIR = "detections"
os.makedirs(SAVE_DIR, exist_ok=True)

enable_alert = st.checkbox("Enable Alerts", True)
target_class = st.selectbox("Select Object for Alert", ["person", "cell phone", "bottle"])

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

last_saved_time = 0
SAVE_COOLDOWN = 5  

def video_frame_callback(frame):
    global last_saved_time

    img = frame.to_ndarray(format="bgr24")

    results = model.track(img, persist=True, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    names = model.names
    counts = {}
    alert_triggered = False

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]

        counts[cls_name] = counts.get(cls_name, 0) + 1

        if enable_alert and cls_name == target_class:
            alert_triggered = True

    y_offset = 30
    for cls_name, count in counts.items():
        cv2.putText(
            annotated_frame,
            f"{cls_name}: {count}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        y_offset += 25

    current_time = time.time()

    if alert_triggered:
        cv2.putText(
            annotated_frame,
            f"⚠ ALERT: {target_class} detected!",
            (10, annotated_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )

        if current_time - last_saved_time > SAVE_COOLDOWN:
            filename = os.path.join(
                SAVE_DIR, f"{target_class}_{int(current_time)}.jpg"
            )
            cv2.imwrite(filename, annotated_frame)
            last_saved_time = current_time

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)